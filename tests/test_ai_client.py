"""Retry behaviour of ask_json with a fake provider (no network, no waiting).

    python tests/test_ai_client.py
"""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer import ai_client  # noqa: E402

BUSY = RuntimeError("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently "
                    "experiencing high demand.', 'status': 'UNAVAILABLE'}}")


def fresh():
    ai_client._gave_up_at.clear()


def test_recovers_after_busy():
    fresh()
    calls = []

    def flaky(system, user, max_tokens):
        calls.append(1)
        if len(calls) < 3:
            raise BUSY
        return '{"ok": true}'
    with mock.patch.dict(ai_client._CALLERS, {"gemini": flaky}), mock.patch.object(ai_client.time, "sleep"):
        assert ai_client.ask_json("s", "u", "gemini") == {"ok": True}
    assert len(calls) == 3
    print("OK recovers on attempt 3")


def test_gives_up_then_fails_fast():
    fresh()
    calls = []

    def down(system, user, max_tokens):
        calls.append(1)
        raise BUSY
    with mock.patch.dict(ai_client._CALLERS, {"gemini": down}), mock.patch.object(ai_client.time, "sleep"):
        try:
            ai_client.ask_json("s", "u", "gemini")
            raise AssertionError("should fail")
        except ai_client.AIError as e:
            assert "overloaded right now (tried 4 times)" in str(e), e
        assert len(calls) == 4
        try:  # second call within the cool-off: no new requests
            ai_client.ask_json("s", "u", "gemini")
            raise AssertionError("should fail fast")
        except ai_client.AIError as e:
            assert "skipped to save time" in str(e), e
        assert len(calls) == 4
    print("OK gives up after 4 tries, then fails fast")


def test_no_retry_on_real_errors():
    fresh()
    calls = []

    def bad_model(system, user, max_tokens):
        calls.append(1)
        raise RuntimeError("404 NOT_FOUND model is no longer available")
    with mock.patch.dict(ai_client._CALLERS, {"gemini": bad_model}), mock.patch.object(ai_client.time, "sleep"):
        try:
            ai_client.ask_json("s", "u", "gemini")
        except ai_client.AIError as e:
            assert "request failed: 404" in str(e)
    assert len(calls) == 1
    print("OK no retry on a wrong model / key")


def test_quota_message():
    fresh()

    def limited(system, user, max_tokens):
        raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")
    with mock.patch.dict(ai_client._CALLERS, {"gemini": limited}), mock.patch.object(ai_client.time, "sleep"):
        try:
            ai_client.ask_json("s", "u", "gemini")
        except ai_client.AIError as e:
            assert "free quota" in str(e)
    print("OK quota message")


if __name__ == "__main__":
    test_recovers_after_busy()
    test_gives_up_then_fails_fast()
    test_no_retry_on_real_errors()
    test_quota_message()
