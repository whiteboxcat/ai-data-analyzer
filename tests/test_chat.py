"""Query engine and "Ask AI" chat, with a fake AI (no network).

    python tests/test_chat.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyzer import ai_client, chat, industry  # noqa: E402
from analyzer.pipeline import run  # noqa: E402
from analyzer.query import QueryError, run_query  # noqa: E402


def analysed(tmp: Path) -> dict:
    with mock.patch.object(industry.requests, "get", side_effect=OSError("offline")):
        r = run(ROOT / "tests/sample_messy.xlsx", with_trends=False, data_out=tmp / "data.pkl")
    r["file"] = "sample_messy.xlsx"
    return r


def test_queries(data):
    sheets, profiles = data
    r = run_query(sheets, profiles, {"sheet": "Sales", "group_by": ["Channel"],
                                     "metrics": [{"column": "GMV", "agg": "sum"}, {"agg": "count"}]})
    assert r["columns"] == ["Channel", "sum of GMV", "rows"] and r["rows"][0][0] == "Shopee", r
    assert sum(row[2] for row in r["rows"]) == 400

    r = run_query(sheets, profiles, {"sheet": "Sales", "group_by": ["Order Date"], "time_grain": "month",
                                     "metrics": [{"column": "Net", "agg": "sum"}]})
    assert r["rows"][0][0] == "2025-01-01" and len(r["rows"]) == 9, r["rows"]    # chronological

    r = run_query(sheets, profiles, {"sheet": "Sales", "filters": [
        {"column": "Order Date", "op": "between", "value": ["2025-03-01", "2025-03-31"]},
        {"column": "channel", "op": "==", "value": "SHOPEE"}]})
    assert r["kind"] == "rows" and r["matched_rows"] > 0
    assert all(row[r["columns"].index("Channel")] == "Shopee" for row in r["rows"])

    r = run_query(sheets, profiles, {"sheet": "Sales", "filters": [{"column": "Net", "op": ">", "value": 5e6}]})
    assert r["matched_rows"] == 1                                                  # the 98M outlier

    for bad, msg in [({"sheet": "Nope"}, "Unknown sheet"),
                     ({"sheet": "Sales", "group_by": ["Colour"]}, "Unknown group_by"),
                     ({"sheet": "Sales", "metrics": [{"column": "City", "agg": "sum"}]}, "isn't numeric"),
                     ({"sheet": "Sales", "filters": [{"column": "City", "op": "~", "value": 1}]}, "Unknown operator")]:
        try:
            run_query(sheets, profiles, bad)
            raise AssertionError(f"should fail: {bad}")
        except QueryError as e:
            assert msg in str(e), (msg, str(e))
    print("OK queries: group, time buckets, filters, outliers, clear errors")


def scripted_ai(script, seen):
    """Fake provider that replays `script` (a list of dicts) and records each payload."""
    def caller(system, user, max_tokens, model=None):
        seen.append(json.loads(user))
        item = script[len(seen) - 1]
        return item if isinstance(item, str) else json.dumps(item)   # str = raw reply text
    return caller


def test_chat_loop(result, data):
    seen = []
    script = [
        {"action": "query", "why": "Bad column first", "query": {"sheet": "Sales", "group_by": ["Chanel"]}},
        {"action": "query", "why": "GMV by channel",
         "query": {"sheet": "Sales", "group_by": ["Channel"], "metrics": [{"column": "GMV", "agg": "sum"}]}},
        {"action": "answer", "answer": "**Shopee** leads GMV.\n- Shopee: 114,260,000"},
    ]
    with mock.patch.dict(ai_client._CALLERS, {"gemini": scripted_ai(script, seen)}):
        out = chat.ask(result, data, [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}],
                       "Which channel sells most?", "gemini")
    assert out["answer"].startswith("**Shopee**")
    assert "error" in out["steps"][0] and "Unknown group_by" in out["steps"][0]["error"]
    assert out["steps"][1]["result"]["rows"][0][0] == "Shopee"
    assert seen[2]["query_results_so_far"][1]["result"]["rows"][0][0] == "Shopee"     # AI saw the numbers
    assert seen[0]["conversation"][1]["role"] == "assistant"
    assert "sample_rows" in seen[0]["dataset"]["sheets"][0] and "rows" not in seen[0]   # summary, not raw data
    print("OK chat loop: bad query -> error fed back -> fixed query -> answer from real numbers")

    seen.clear()
    endless = [{"action": "query", "why": "again", "query": {"sheet": "Sales"}}] * 3 + [{"action": "answer", "answer": "Done."}]
    with mock.patch.dict(ai_client._CALLERS, {"gemini": scripted_ai(endless, seen)}):
        out = chat.ask(result, data, [], "loop?", "gemini")
    assert len(seen) == chat.MAX_STEPS and seen[-1]["instruction"].startswith("No more queries")
    print(f"OK chat stops after {chat.MAX_STEPS} AI calls")


def test_non_json_replies(result, data):
    seen = []
    prose = "While anomalies alone don't prove fraud, there are **risk signals**:\n- 62 projects with Unknown contractors"
    with mock.patch.dict(ai_client._CALLERS, {"gemini": scripted_ai([prose], seen)}):
        out = chat.ask(result, data, [], "Any corruption?", "gemini")
    assert out["answer"] == prose
    assert "never state or imply" in chat.SYSTEM

    seen.clear()
    script = ['{"action": "query", "query": {"sheet": "Sales", "group_by": ["Ch',     # cut off
              {"action": "answer", "answer": "Recovered."}]
    with mock.patch.dict(ai_client._CALLERS, {"gemini": scripted_ai(script, seen)}):
        out = chat.ask(result, data, [], "Which channel?", "gemini")
    assert out["answer"] == "Recovered." and "not valid JSON" in seen[1]["query_results_so_far"][0]["error"]

    seen.clear()
    cut_answer = '{"action": "answer", "answer": "Shopee leads.\\n- Shopee: 114,260,000\\n- TikT'
    with mock.patch.dict(ai_client._CALLERS, {"gemini": scripted_ai([cut_answer], seen)}):
        out = chat.ask(result, data, [], "Which channel?", "gemini")
    assert out["answer"].startswith("Shopee leads.\n- Shopee: 114,260,000") and "cut off" in out["answer"], out
    print("OK prose replies, cut-off answers and broken queries are handled")


def test_endpoint(result, tmp):
    os.environ.update({"APP_PASSWORD": "", "GEMINI_API_KEY": "x"})
    import app as A
    job = "abcdef123456"
    d = A.UPLOADS / job
    d.mkdir(parents=True, exist_ok=True)
    try:
        (d / "result.json").write_text(json.dumps(result))
        shutil.copy(tmp / "data.pkl", d / "data.pkl")
        c = A.app.test_client()
        page = c.get(f"/report/{job}").get_data(as_text=True)
        assert 'id="ask"' in page and "Ask about this data" in page and 'data-download="png"' in page
        seen = []
        script = [{"action": "answer", "answer": "Fine."}]
        with mock.patch.dict(ai_client._CALLERS, {"gemini": scripted_ai(script, seen)}):
            ok = c.post(f"/report/{job}/chat", json={"message": "Hi", "history": [], "provider": "gemini"})
        assert ok.status_code == 200 and ok.get_json()["answer"] == "Fine.", ok.get_data(as_text=True)
        assert c.post(f"/report/{job}/chat", data={"message": "x"}).status_code == 415     # form posts refused
        assert c.post(f"/report/{job}/chat", json={"message": "x", "provider": "openai"}).status_code == 400
        assert c.post(f"/report/{job}/chat", json={"message": " ", "provider": "gemini"}).status_code == 400
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("OK endpoint: page has the tab and download buttons; JSON only; provider and empty checks")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    try:
        result = analysed(tmp)
        data = chat.load_data(tmp / "data.pkl")
        test_queries(data)
        test_chat_loop(result, data)
        test_non_json_replies(result, data)
        test_endpoint(result, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
