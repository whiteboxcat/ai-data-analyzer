"""Offline end-to-end test: fake AI + fake news feed, no API keys or internet needed.

    python tests/make_sample.py
    python tests/test_pipeline.py
"""
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyzer import ai_client, industry  # noqa: E402
from analyzer.pipeline import run  # noqa: E402

FAKE_PLAN = {
    "workbook_summary": "Marketplace fashion sales for Jan-Sep 2025 with daily traffic and a product list.",
    "sheets": [
        {"sheet": "Sales", "description": "One row per order line.",
         "charts": [
             {"type": "line", "x": "Order Date", "y": "Net", "agg": "sum", "time_grain": "month", "title": "Monthly net sales"},
             {"type": "bar", "x": "Channel", "y": "Net", "agg": "sum", "group_by": "Category", "title": "Net by channel & category"},
             {"type": "pie", "x": "Category", "y": "GMV", "agg": "sum", "title": "GMV share by category"},
             {"type": "line", "x": "Order No", "y": "GMV", "title": "INVALID: id on x"},
             {"type": "bar", "x": "City", "y": "Category", "agg": "sum", "title": "INVALID: sum text"},
         ]},
        {"sheet": "Daily Traffic", "description": "Daily web traffic.",
         "charts": [{"type": "scatter", "x": "Ad Spend", "y": "Orders", "title": "Ad spend vs orders"}]},
    ],
    "industry": {"name": "Fashion e-commerce", "market": "Indonesia", "confidence": 0.85,
                 "likely_competitors": ["Erigo", "Zalora"], "news_queries": ["fashion e-commerce Indonesia"]},
}
FAKE_INSIGHTS = {
    "executive_summary": "Net sales peaked in May.", "chart_insights": {"Sales::0": "Peak in May."},
    "insights": [{"title": "Shopee leads", "detail": "42% of qty", "sheet": "Sales"}],
    "recommendations": [{"title": "Double down on Shopee", "detail": "...", "priority": "high", "based_on": "channel mix"}],
    "data_caveats": [], "next_questions": [],
}
FAKE_TRENDS = {"industry_overview": "Social commerce is growing.",
               "trends": [{"title": "Live shopping", "detail": "...", "refs": [1]}],
               "competitor_moves": [], "implications": [], "watchlist": []}

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Live shopping boom in Indonesia - Example News</title><link>https://example.com/a</link>
<pubDate>Mon, 15 Sep 2025 08:00:00 GMT</pubDate><source url="https://example.com">Example News</source></item>
</channel></rss>"""


def fake_call(system, user, max_tokens):
    if "industry analyst" in system:
        return json.dumps(FAKE_TRENDS)
    if "busy manager" in system:
        return "```json\n" + json.dumps(FAKE_INSIGHTS) + "\n```"   # also tests fence stripping
    return json.dumps(FAKE_PLAN)


def main():
    fake_resp = mock.Mock(content=RSS, raise_for_status=lambda: None)
    with mock.patch.dict(ai_client._CALLERS, {"claude": fake_call}), \
         mock.patch.object(industry.requests, "get", return_value=fake_resp):
        r = run(ROOT / "tests/sample_messy.xlsx", provider="claude",
                context={"market": "Indonesia"}, cleaned_out=ROOT / "tests/_cleaned.xlsx")

    sales = next(s for s in r["sheets"] if s["sheet"] == "Sales")
    titles = [c["title"] for c in sales["charts"]]
    assert "Monthly net sales" in titles, titles
    assert not any("INVALID" in t for t in titles), "invalid charts must be rejected"
    assert len(r["rejected_charts"]) == 2, r["rejected_charts"]
    assert r["insights"]["executive_summary"]
    assert r["trends"]["status"] == "ok" and r["trends"]["articles"][0]["source"] == "Example News"
    assert r["trends"]["summary"]["trends"]
    # dates: Jan-Sep only (dayfirst parsing must not flip ISO dates)
    k = next(k for k in sales["kpis"] if "range" in k["label"])
    assert k["value"].startswith("2025-01") and "2025-09" in k["sub"], k
    # GMV text converted and summed
    assert any(k["label"] == "Total GMV" and k["value"] > 1e8 for k in sales["kpis"])
    json.dumps(r)
    print("OK - charts:", titles)
    print("rejected:", [x["reason"] for x in r["rejected_charts"]])

    # no-AI path still works
    with mock.patch.object(industry.requests, "get", side_effect=OSError("offline")):
        r2 = run(ROOT / "tests/sample_sales.csv", provider=None)
    assert r2["sheets"][0]["charts"] and r2["trends"]["status"] == "no_news"
    print("OK - fallback path, CSV charts:", [c["title"] for c in r2["sheets"][0]["charts"]])


if __name__ == "__main__":
    main()
