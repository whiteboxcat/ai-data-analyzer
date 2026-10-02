"""Latest industry trends, media coverage and competitor news.

1. The planner (AI) detects the industry, market and likely competitors.
2. We pull REAL recent headlines from Google News RSS (free, no key).
3. The AI summarises only those headlines into trends + competitor moves,
   citing them — so "latest" means actually latest, not the model's memory.
"""
from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests

from .ai_client import AIError, ask_json

# market -> Google News locale (hl, gl, ceid)
LOCALES = {
    "indonesia": ("id", "ID", "ID:id"), "malaysia": ("en-MY", "MY", "MY:en"),
    "singapore": ("en-SG", "SG", "SG:en"), "philippines": ("en-PH", "PH", "PH:en"),
    "vietnam": ("vi", "VN", "VN:vi"), "thailand": ("th", "TH", "TH:th"),
    "india": ("en-IN", "IN", "IN:en"), "australia": ("en-AU", "AU", "AU:en"),
    "united kingdom": ("en-GB", "GB", "GB:en"), "uk": ("en-GB", "GB", "GB:en"),
}
DEFAULT_LOCALE = ("en-US", "US", "US:en")
TIMEOUT = 8


def _locale(market: str | None):
    m = (market or "").lower()
    return next((v for k, v in LOCALES.items() if k in m), DEFAULT_LOCALE)


def fetch_news(query: str, market: str | None = None, days: int = 60, limit: int = 8) -> list[dict]:
    hl, gl, ceid = _locale(market)
    url = (f"https://news.google.com/rss/search?q={quote_plus(query)}+when:{days}d"
           f"&hl={hl}&gl={gl}&ceid={ceid}")
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0 ai-data-analyzer"})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    items = []
    for it in root.iter("item"):
        title = html.unescape(it.findtext("title") or "")
        source = it.findtext("source") or ""
        if source and title.endswith(" - " + source):
            title = title[: -len(source) - 3]
        try:
            published = parsedate_to_datetime(it.findtext("pubDate")).date().isoformat()
        except Exception:
            published = None
        items.append({"title": title, "source": source, "url": it.findtext("link"),
                      "published": published, "query": query})
        if len(items) >= limit:
            break
    return items


def gather(industry: dict) -> tuple[list[dict], list[str]]:
    """Fetch headlines for the industry queries and each competitor in parallel."""
    market = industry.get("market")
    queries = list(industry.get("news_queries") or [])[:4]
    if not queries and industry.get("name"):
        queries = [f"{industry['name']} trends" + (f" {market}" if market else "")]
    comp_queries = [f'"{c}"' for c in (industry.get("likely_competitors") or [])[:5]]
    if industry.get("own_brand"):
        comp_queries.insert(0, f'"{industry["own_brand"]}"')

    jobs = [(q, "industry") for q in queries] + [(q, "competitor") for q in comp_queries]
    articles, errors, seen = [], [], set()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fetch_news, q, market, 60, 6 if kind == "industry" else 4): (q, kind)
                   for q, kind in jobs}
        for f, (q, kind) in futures.items():
            try:
                for a in f.result():
                    key = re.sub(r"\W+", "", a["title"].lower())[:80]
                    if key not in seen:
                        seen.add(key)
                        articles.append({**a, "kind": kind})
            except Exception as e:  # network blocked, timeout ...
                errors.append(f"{q}: {e.__class__.__name__}")
    articles.sort(key=lambda a: a["published"] or "", reverse=True)
    for i, a in enumerate(articles):
        a["ref"] = i + 1
    return articles, errors


TREND_SYSTEM = """You are an industry analyst. You get: the detected industry/market, the company's own data
highlights, and a numbered list of RECENT news headlines (with source and date).
Use ONLY those headlines for claims about the market. Cite headline numbers like [3]. If headlines are thin or
off-topic, say so instead of inventing. {language_rule}
Return JSON:
{{
  "industry_overview": "2-3 sentences on the current state of this industry from the headlines",
  "trends": [{{"title": "trend", "detail": "what is happening", "refs": [1, 4]}}],
  "competitor_moves": [{{"competitor": "name", "move": "what they did", "refs": [2]}}],
  "implications": [{{"title": "what it means for this business", "detail": "connect the trend to their own numbers"}}],
  "watchlist": ["things to monitor next"]
}}"""


def summarize(industry: dict, articles: list[dict], data_highlights: dict, provider: str,
              language: str | None = None) -> dict:
    rule = f"Write in {language}." if language else "Write in English."
    user = json.dumps({
        "industry": industry,
        "own_data_highlights": data_highlights,
        "headlines": [{"ref": a["ref"], "kind": a["kind"], "title": a["title"],
                       "source": a["source"], "date": a["published"]} for a in articles[:40]],
    }, ensure_ascii=False, default=str)
    return ask_json(TREND_SYSTEM.format(language_rule=rule), user, provider, max_tokens=8000)


def run(industry: dict | None, provider: str | None, data_highlights: dict, language: str | None = None,
        enabled: bool = True) -> dict:
    if not enabled:
        return {"status": "disabled"}
    if not industry or not industry.get("name"):
        return {"status": "no_industry",
                "message": "Could not detect an industry from this data. Enter it on the upload form to get trends."}
    articles, errors = gather(industry)
    out = {"status": "ok", "industry": industry, "articles": articles, "fetch_errors": errors}
    if not articles:
        out["status"] = "no_news"
        out["message"] = "No news could be fetched" + (f" ({errors[0]})" if errors else "") + "."
        return out
    if provider:
        try:
            out["summary"] = summarize(industry, articles, data_highlights, provider, language)
        except AIError as e:
            out["summary_error"] = str(e)
    return out
