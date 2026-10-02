"""One interface for OpenAI (GPT), Anthropic (Claude) and Google (Gemini).

    from analyzer.ai_client import ask_json, available_providers
    data = ask_json(system_prompt, user_prompt, provider="claude")

Keys and models come from .env:
    OPENAI_API_KEY      OPENAI_MODEL     (default gpt-5-mini)
    ANTHROPIC_API_KEY   ANTHROPIC_MODEL  (default claude-sonnet-5-5)
    GEMINI_API_KEY      GEMINI_MODEL     (default gemini-2.5-flash)

Reasoning models (gpt-5, gemini-2.5) spend part of max_tokens on thinking,
so limits are generous. SDKs are imported lazily, so you only need to `pip install` the ones you use.
"""
from __future__ import annotations

import json
import os
import re

PROVIDERS = {
    "openai": {"label": "ChatGPT (OpenAI)", "key": "OPENAI_API_KEY",
               "model_env": "OPENAI_MODEL", "default_model": "gpt-5-mini"},
    "claude": {"label": "Claude (Anthropic)", "key": "ANTHROPIC_API_KEY",
               "model_env": "ANTHROPIC_MODEL", "default_model": "claude-sonnet-5-5"},
    "gemini": {"label": "Gemini (Google)", "key": "GEMINI_API_KEY",
               "model_env": "GEMINI_MODEL", "default_model": "gemini-2.5-flash"},
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


def ask_json(system: str, user: str, provider: str, max_tokens: int = 8000) -> dict:
    """Send a prompt and return the parsed JSON object the model replies with."""
    if provider not in _CALLERS:
        raise AIError(f"Unknown provider '{provider}'")
    system = system + "\n\nRespond with ONE valid JSON object only. No markdown, no commentary."
    try:
        raw = _CALLERS[provider](system, user, max_tokens)
    except ImportError as e:
        raise AIError(f"SDK for {provider} not installed: {e}. See requirements.txt") from e
    except KeyError as e:
        raise AIError(f"Missing API key {e} in .env") from e
    except Exception as e:  # network, auth, rate limit ...
        raise AIError(f"{PROVIDERS[provider]['label']} request failed: {e}") from e
    return _extract_json(raw)
