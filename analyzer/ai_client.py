"""One interface for OpenAI (GPT), Anthropic (Claude) and Google (Gemini).

    from analyzer.ai_client import ask_json, available_providers
    data = ask_json(system_prompt, user_prompt, provider="claude")

Keys and models come from .env:
    OPENAI_API_KEY      OPENAI_MODEL     (default gpt-5-mini)
    ANTHROPIC_API_KEY   ANTHROPIC_MODEL  (default claude-sonnet-5-5)
    GEMINI_API_KEY      GEMINI_MODEL     (default gemini-3.8-flash)

Reasoning models (gpt-5, gemini-2.5) spend part of max_tokens on thinking,
so limits are generous. SDKs are imported lazily, so you only need to `pip install` the ones you use.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time

log = logging.getLogger(__name__)

PROVIDERS = {
    "openai": {"label": "ChatGPT (OpenAI)", "key": "OPENAI_API_KEY",
               "model_env": "OPENAI_MODEL", "default_model": "gpt-5-mini"},
    "claude": {"label": "Claude (Anthropic)", "key": "ANTHROPIC_API_KEY",
               "model_env": "ANTHROPIC_MODEL", "default_model": "claude-sonnet-5-5"},
    "gemini": {"label": "Gemini (Google)", "key": "GEMINI_API_KEY",
               "model_env": "GEMINI_MODEL", "default_model": "gemini-3.8-flash"},
}


class AIError(RuntimeError):
    pass


def available_providers() -> list[dict]:
    """Providers whose API key is set in the environment."""
    out = []
    for pid, cfg in PROVIDERS.items():
        key = os.getenv(cfg["key"]) or (pid == "gemini" and os.getenv("GOOGLE_API_KEY"))
        if key:
            out.append({"id": pid, "label": cfg["label"], "model": _model(pid)})
    return out


def _model(pid: str) -> str:
    cfg = PROVIDERS[pid]
    return (os.getenv(cfg["model_env"]) or "").strip() or cfg["default_model"]


def _extract_json(text: str) -> dict:
    """Parse JSON even if the model wrapped it in ```json fences or prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise AIError(f"AI did not return valid JSON: {text[:200]}")


# --------------------------------------------------------------------------- #
# Provider calls. `model` lets ask_json switch to a fallback model.
# --------------------------------------------------------------------------- #
def _call_openai(system: str, user: str, max_tokens: int, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        max_completion_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _call_claude(system: str, user: str, max_tokens: int, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _gemini_client():
    from google import genai
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _call_gemini(system: str, user: str, max_tokens: int, model: str) -> str:
    from google.genai import types
    # Keep `client` in a variable: google-genai closes its connection as soon as
    # the Client object is garbage-collected, even mid-request.
    client = _gemini_client()
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            max_output_tokens=max_tokens,
        ),
    )
    return resp.text or ""


_CALLERS = {"openai": _call_openai, "claude": _call_claude, "gemini": _call_gemini}


# --------------------------------------------------------------------------- #
# Fallback models
# --------------------------------------------------------------------------- #
# Set e.g. GEMINI_FALLBACK_MODELS=gemini-x-flash-lite,gemini-x-pro to choose them yourself.
# For Gemini, if it isn't set, the app asks Google which models this key can use.
_SKIP_MODEL = re.compile(r"embed|tts|image|audio|live|native|vision|computer|robotics|aqa|learnlm|gemma", re.I)
_discovered: dict[str, tuple[float, list[str]]] = {}
DISCOVERY_TTL = 6 * 3600
MAX_FALLBACKS = 3


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def rank_gemini_models(names: list[str]) -> list[str]:
    """Best fallbacks first: text models, Flash before Pro, stable before preview, newest first."""
    usable = [n for n in names if n.startswith("gemini-") and not _SKIP_MODEL.search(n)]
    return sorted(dict.fromkeys(usable), key=lambda n: (
        0 if "flash" in n else 1,
        1 if re.search(r"preview|exp|latest", n) else 0,
        -_version(n),
        len(n),
    ))


def _discover_gemini_models() -> list[str]:
    cached = _discovered.get("gemini")
    if cached and time.time() - cached[0] < DISCOVERY_TTL:
        return cached[1]
    names = []
    try:
        client = _gemini_client()          # must outlive the paged iteration below
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if actions and "generateContent" not in actions:
                continue
            names.append(str(m.name).removeprefix("models/"))
    except Exception as e:  # noqa: BLE001 - discovery is best effort
        log.warning("Could not list Gemini models: %s", str(e)[:150])
    ranked = rank_gemini_models(names)
    _discovered["gemini"] = (time.time(), ranked)
    return ranked


def _fallback_models(provider: str, exclude: set[str]) -> list[str]:
    env = os.getenv(f"{PROVIDERS[provider]['model_env'].rsplit('_', 1)[0]}_FALLBACK_MODELS", "")
    names = [n.strip() for n in env.split(",") if n.strip()]
    if not names and provider == "gemini":
        names = _discover_gemini_models()
    return [n for n in names if n not in exclude][:MAX_FALLBACKS]


# --------------------------------------------------------------------------- #
# Calling with retries and fallbacks
# --------------------------------------------------------------------------- #
# Temporary problems worth retrying: overloaded servers, rate limits, network blips.
_BUSY = re.compile(r"\b(500|502|503|504)\b|UNAVAILABLE|overloaded|high demand|"
                   r"timed? ?out|connection (?:error|reset|aborted)", re.I)
_LIMIT = re.compile(r"\b429\b|RESOURCE_EXHAUSTED|rate.?limit|quota", re.I)
RETRY_WAITS = (3, 8, 20)          # main model: 4 attempts in total
FALLBACK_WAITS = (3,)             # each fallback model: 2 attempts
COOL_OFF = 90                     # after everything failed, fail fast for this long
STICKY = 15 * 60                  # keep using a fallback that worked for this long
_gave_up_at: dict[str, float] = {}
_sticky: dict[str, tuple[str, float]] = {}
last_model: dict[str, str] = {}   # model that answered the last successful call


def _temporary(err: Exception) -> bool:
    text = str(err)
    return bool(_BUSY.search(text) or _LIMIT.search(text))


def _friendly(label: str, err: Exception, tried: list[str]) -> str:
    text = str(err)
    short = text if len(text) < 300 else text[:300] + "…"
    models = ", ".join(tried)
    if _LIMIT.search(text):
        return (f"{label} rate limit or free quota reached (tried {models}). Wait a minute and try again; "
                f"if it keeps happening, the daily free quota is used up. Details: {short}")
    if _BUSY.search(text):
        return (f"{label} is overloaded right now (tried {models}). This is on the provider's side and "
                f"usually passes within minutes; try again shortly. Details: {short}")
    return f"{label} request failed ({models}): {short}"


def _attempt(provider: str, model: str, system: str, user: str, max_tokens: int, waits) -> str:
    for attempt in range(len(waits) + 1):
        try:
            return _CALLERS[provider](system, user, max_tokens, model)
        except (ImportError, KeyError):
            raise
        except Exception as e:  # network, auth, rate limit ...
            if _temporary(e) and attempt < len(waits):
                wait = waits[attempt] + random.uniform(0, 1.5)
                log.warning("%s %s attempt %d failed (%s); retrying in %.0fs", provider, model,
                            attempt + 1, str(e)[:120], wait)
                time.sleep(wait)
                continue
            raise
    raise AssertionError("unreachable")


def ask_json(system: str, user: str, provider: str, max_tokens: int = 8000) -> dict:
    """Send a prompt and return the parsed JSON object the model replies with.

    1. Temporary errors (overloaded, rate limited, network) are retried with waits.
    2. If the main model stays unavailable, other models of the same provider are
       tried (for Gemini: the ones Google lists for this key). One that works is
       kept for STICKY seconds so later calls go straight to it.
    3. If everything fails, calls fail fast for COOL_OFF seconds so one analysis
       doesn't wait three times over.
    """
    if provider not in _CALLERS:
        raise AIError(f"Unknown provider '{provider}'")
    label = PROVIDERS[provider]["label"]
    since = time.time() - _gave_up_at.get(provider, 0)
    if since < COOL_OFF:
        raise AIError(f"{label} was unavailable a moment ago; skipped to save time. Try again in "
                      f"{int(COOL_OFF - since) + 1} seconds.")

    default = _model(provider)
    sticky = _sticky.get(provider)
    first = sticky[0] if sticky and time.time() - sticky[1] < STICKY else default
    system = system + "\n\nRespond with ONE valid JSON object only. No markdown, no commentary."
    tried: list[str] = []
    try:
        tried.append(first)
        raw, used = _attempt(provider, first, system, user, max_tokens, RETRY_WAITS), first
    except ImportError as e:
        raise AIError(f"SDK for {provider} not installed: {e}. See requirements.txt") from e
    except KeyError as e:
        raise AIError(f"Missing API key {e} in .env") from e
    except Exception as e:  # noqa: BLE001
        if not _temporary(e):
            raise AIError(_friendly(label, e, tried)) from e
        raw, used, last_err = None, None, e
        for alt in _fallback_models(provider, exclude={first, default}):
            tried.append(alt)
            log.warning("%s %s unavailable; trying %s", label, first, alt)
            try:
                raw, used = _attempt(provider, alt, system, user, max_tokens, FALLBACK_WAITS), alt
                break
            except Exception as e2:  # noqa: BLE001 - try the next model
                last_err = e2
        if raw is None:
            _gave_up_at[provider] = time.time()
            raise AIError(_friendly(label, last_err if _temporary(last_err) else e, tried)) from e

    _gave_up_at.pop(provider, None)
    if used != default:
        _sticky[provider] = (used, time.time())
    else:
        _sticky.pop(provider, None)
    last_model[provider] = used
    return _extract_json(raw)
