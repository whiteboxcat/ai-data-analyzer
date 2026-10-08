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
    return os.getenv(cfg["model_env"], cfg["default_model"])


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
# Provider calls
# --------------------------------------------------------------------------- #
def _call_openai(system: str, user: str, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=_model("openai"),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        max_completion_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _call_claude(system: str, user: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=_model("claude"),
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _call_gemini(system: str, user: str, max_tokens: int) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model=_model("gemini"),
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            max_output_tokens=max_tokens,
        ),
    )
    return resp.text or ""


_CALLERS = {"openai": _call_openai, "claude": _call_claude, "gemini": _call_gemini}


# Temporary problems worth retrying: overloaded servers, rate limits, network blips.
_BUSY = re.compile(r"\b(500|502|503|504)\b|UNAVAILABLE|overloaded|high demand|"
                   r"timed? ?out|connection (?:error|reset|aborted)", re.I)
_LIMIT = re.compile(r"\b429\b|RESOURCE_EXHAUSTED|rate.?limit|quota", re.I)
RETRY_WAITS = (3, 8, 20)          # seconds between attempts (4 attempts in total)
COOL_OFF = 90                     # after giving up, fail fast for this long
_gave_up_at: dict[str, float] = {}


def _friendly(label: str, err: Exception, attempts: int) -> str:
    text = str(err)
    short = text if len(text) < 300 else text[:300] + "…"
    if _LIMIT.search(text):
        return (f"{label} rate limit or free quota reached (tried {attempts} times). Wait a minute and "
                f"try again; if it keeps happening, the daily free quota is used up. Details: {short}")
    if _BUSY.search(text):
        return (f"{label} is overloaded right now (tried {attempts} times). This is on the provider's "
                f"side and usually passes within minutes; try again shortly. Details: {short}")
    return f"{label} request failed: {short}"


def ask_json(system: str, user: str, provider: str, max_tokens: int = 8000) -> dict:
    """Send a prompt and return the parsed JSON object the model replies with.

    Temporary errors (overloaded, rate limited, network) are retried with
    increasing waits. After giving up, further calls to the same provider fail
    fast for a short while, so one analysis doesn't wait three times over.
    """
    if provider not in _CALLERS:
        raise AIError(f"Unknown provider '{provider}'")
    label = PROVIDERS[provider]["label"]
    since = time.time() - _gave_up_at.get(provider, 0)
    if since < COOL_OFF:
        raise AIError(f"{label} was unavailable a moment ago; skipped to save time. Try again in "
                      f"{int(COOL_OFF - since) + 1} seconds.")

    system = system + "\n\nRespond with ONE valid JSON object only. No markdown, no commentary."
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            raw = _CALLERS[provider](system, user, max_tokens)
            break
        except ImportError as e:
            raise AIError(f"SDK for {provider} not installed: {e}. See requirements.txt") from e
        except KeyError as e:
            raise AIError(f"Missing API key {e} in .env") from e
        except Exception as e:  # network, auth, rate limit ...
            temporary = bool(_BUSY.search(str(e)) or _LIMIT.search(str(e)))
            if temporary and attempt < len(RETRY_WAITS):
                wait = RETRY_WAITS[attempt] + random.uniform(0, 1.5)
                log.warning("%s attempt %d failed (%s); retrying in %.0fs", label, attempt + 1,
                            str(e)[:120], wait)
                time.sleep(wait)
                continue
            if temporary:
                _gave_up_at[provider] = time.time()
            raise AIError(_friendly(label, e, attempt + 1)) from e
    _gave_up_at.pop(provider, None)
    return _extract_json(raw)
