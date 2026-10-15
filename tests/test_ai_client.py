"""Retries and model fallback in ask_json, with a fake provider (no network, no waiting).

    python tests/test_ai_client.py
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer import ai_client  # noqa: E402

BUSY = RuntimeError("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently "
                    "experiencing high demand.', 'status': 'UNAVAILABLE'}}")
DEFAULT = ai_client.PROVIDERS["gemini"]["default_model"]


def fresh():
    ai_client._gave_up_at.clear()
    ai_client._sticky.clear()
    ai_client._discovered.clear()
    ai_client.last_model.clear()


def patched(caller, fallbacks=()):
    """Fake Gemini caller, no sleeping, fixed fallback list, no env overrides."""
    return [mock.patch.dict(ai_client._CALLERS, {"gemini": caller}),
            mock.patch.object(ai_client.time, "sleep"),
            mock.patch.object(ai_client, "_discover_gemini_models", return_value=list(fallbacks)),
            mock.patch.dict(os.environ, {"GEMINI_MODEL": "", "GEMINI_FALLBACK_MODELS": ""})]


def run_with(patches, fn):
    for p in patches:
        p.start()
    try:
        return fn()
    finally:
        for p in reversed(patches):
            p.stop()


def test_recovers_after_busy():
    fresh()
    calls = []

    def flaky(system, user, max_tokens, model):
        calls.append(model)
        if len(calls) < 3:
            raise BUSY
        return '{"ok": true}'
    assert run_with(patched(flaky), lambda: ai_client.ask_json("s", "u", "gemini")) == {"ok": True}
    assert calls == [DEFAULT] * 3 and ai_client.last_model["gemini"] == DEFAULT
    print("OK recovers on attempt 3 of the main model")


def test_falls_back_to_another_model_and_sticks():
    fresh()
    calls = []

    def main_down(system, user, max_tokens, model):
        calls.append(model)
        if model == DEFAULT or model == "gemini-a-pro":
            raise BUSY
        return '{"ok": "%s"}' % model
    patches = patched(main_down, fallbacks=["gemini-a-pro", "gemini-b-flash-lite"])
    out = run_with(patches, lambda: ai_client.ask_json("s", "u", "gemini"))
    assert out == {"ok": "gemini-b-flash-lite"}, out
    assert calls == [DEFAULT] * 4 + ["gemini-a-pro"] * 2 + ["gemini-b-flash-lite"], calls
    calls.clear()
    run_with(patched(main_down, fallbacks=["gemini-a-pro", "gemini-b-flash-lite"]),
             lambda: ai_client.ask_json("s", "u", "gemini"))
    assert calls == ["gemini-b-flash-lite"], calls      # next call goes straight to the working model
    print("OK falls back to gemini-b-flash-lite, then keeps using it")


def test_gives_up_then_fails_fast():
    fresh()
    calls = []

    def down(system, user, max_tokens, model):
        calls.append(model)
        raise BUSY
    patches = patched(down, fallbacks=["gemini-x-flash"])

    def go():
        try:
            ai_client.ask_json("s", "u", "gemini")
            raise AssertionError("should fail")
        except ai_client.AIError as e:
            assert "overloaded right now" in str(e) and "gemini-x-flash" in str(e), e
        n = len(calls)
        try:
            ai_client.ask_json("s", "u", "gemini")
            raise AssertionError("should fail fast")
        except ai_client.AIError as e:
            assert "skipped to save time" in str(e), e
        assert len(calls) == n
    run_with(patches, go)
    assert calls == [DEFAULT] * 4 + ["gemini-x-flash"] * 2, calls
    print("OK tries main + fallback, then fails fast")


def test_no_retry_or_fallback_on_real_errors():
    fresh()
    calls = []

    def bad_key(system, user, max_tokens, model):
        calls.append(model)
        raise RuntimeError("400 INVALID_ARGUMENT API key not valid")

    def go():
        try:
            ai_client.ask_json("s", "u", "gemini")
        except ai_client.AIError as e:
            assert "request failed" in str(e)
    run_with(patched(bad_key, fallbacks=["gemini-x-flash"]), go)
    assert calls == [DEFAULT]
    print("OK no retry or fallback on a wrong key")


def test_env_fallbacks_and_ranking():
    fresh()
    with mock.patch.dict(os.environ, {"GEMINI_FALLBACK_MODELS": "gemini-m1, gemini-m2"}):
        assert ai_client._fallback_models("gemini", {DEFAULT}) == ["gemini-m1", "gemini-m2"]
    ranked = ai_client.rank_gemini_models([
        "gemini-2.0-pro", "gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.9-flash-preview",
        "gemini-3.8-pro", "text-embedding-004", "gemini-embedding-001", "gemini-3.8-flash-image",
        "gemini-3.8-flash-tts", "gemma-3-27b-it"])
    assert ranked == ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.9-flash-preview",
                      "gemini-3.8-pro", "gemini-2.0-pro"], ranked
    print("OK fallback list from env; ranking:", ranked)


def test_discovery_reads_google_list():
    fresh()
    models = [SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["generateContent"]),
              SimpleNamespace(name="models/gemini-3.8-flash-lite", supported_actions=["generateContent"]),
              SimpleNamespace(name="models/gemini-embedding-001", supported_actions=["embedContent"]),
              SimpleNamespace(name="models/gemini-3.8-pro", supported_actions=None)]
    fake = SimpleNamespace(models=SimpleNamespace(list=lambda: iter(models)))
    with mock.patch.object(ai_client, "_gemini_client", return_value=fake):
        assert ai_client._discover_gemini_models() == ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.8-pro"]
    print("OK discovery keeps text models that support generateContent")


if __name__ == "__main__":
    test_recovers_after_busy()
    test_falls_back_to_another_model_and_sticks()
    test_gives_up_then_fails_fast()
    test_no_retry_or_fallback_on_real_errors()
    test_env_fallbacks_and_ranking()
    test_discovery_reads_google_list()


# --------------------------------------------------------------------------- #
# Regression: google-genai closes its HTTP connection when the Client object is
# garbage-collected. Using `_gemini_client().models...` without keeping the
# client failed with "Cannot send a request, as the client has been closed."
# --------------------------------------------------------------------------- #
class _Api:
    closed = False


class _Models:
    def __init__(self, api):
        self.api = api                      # like the real SDK: no reference back to Client

    def _check(self):
        if self.api.closed:
            raise RuntimeError("Cannot send a request, as the client has been closed.")

    def generate_content(self, **kw):
        self._check()
        return SimpleNamespace(text='{"ok": true}')

    def list(self):
        self._check()
        for name in ["models/gemini-3.8-flash", "models/gemini-3.8-flash-lite"]:
            self._check()
            yield SimpleNamespace(name=name, supported_actions=["generateContent"])


class _ClosingClient:
    def __init__(self):
        self._api = _Api()
        self.models = _Models(self._api)

    def __del__(self):
        self._api.closed = True


def _fake_genai_types():
    types = SimpleNamespace(GenerateContentConfig=lambda **kw: kw)
    genai = SimpleNamespace(types=types)
    return {"google": SimpleNamespace(genai=genai), "google.genai": genai, "google.genai.types": types}


def test_client_kept_alive_during_calls():
    fresh()
    with mock.patch.dict(sys.modules, _fake_genai_types()), \
            mock.patch.object(ai_client, "_gemini_client", side_effect=lambda: _ClosingClient()):
        assert ai_client._call_gemini("s", "u", 100, "gemini-3.8-flash") == '{"ok": true}'
        assert ai_client._discover_gemini_models() == ["gemini-3.8-flash", "gemini-3.8-flash-lite"]
    print("OK Gemini client stays open for generate_content and models.list")


if __name__ == "__main__":
    test_client_kept_alive_during_calls()
