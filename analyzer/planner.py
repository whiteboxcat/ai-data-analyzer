"""AI chart planning (+ industry detection) with a small no-AI fallback.

The AI sees the *profile* (column roles, stats, a few sample rows) — never the
full dataset. It answers with structured JSON; every chart it proposes is
validated by Python before anything is drawn.
"""
from __future__ import annotations

import json
import re

from .ai_client import AIError, ask_json
from .charts import validate_spec

# Most important first: money > volume > everything else
PRIORITY_METRICS = [r"revenue|pendapatan|omzet|sales|penjualan|gmv", r"\bnet\b|profit|laba", r"total|amount|nilai",
                    r"orders|qty|quantity|jumlah", r"spend|cost|biaya", r"visitors|clicks|traffic"]
NON_ADDITIVE = re.compile(r"(price|harga|rate|rating|score|pct|percent|persen|margin|avg|average|rata|ratio|age|umur|stock|stok|balance|saldo)", re.I)
MAX_CHARTS_PER_SHEET = 6


def is_additive(metric: str) -> bool:
    """Totals make sense for GMV/Qty, not for Price/Rating/Stock levels."""
    return not NON_ADDITIVE.search(metric)


def rank_metrics(profile: dict) -> list[str]:
    metrics = profile["by_role"].get("metric", [])

    def rank(m):
        for i, pat in enumerate(PRIORITY_METRICS):
            if re.search(pat, m, re.I):
                return (i, metrics.index(m))
        return (len(PRIORITY_METRICS), metrics.index(m))
    return sorted(metrics, key=rank)


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #
def compact_profile(profile: dict) -> dict:
    """Trim the profile so the prompt stays small (and cheap)."""
    return {
        "sheet": profile["sheet"], "rows": profile["rows"],
        "columns": [{k: v for k, v in c.items() if k in ("name", "role", "missing_pct", "unique", "stats", "note")}
                    for c in profile["columns"]],
        "sample_rows": profile["sample_rows"][:5],
    }


PLAN_SYSTEM = """You are a senior data analyst. You receive a PROFILE of a spreadsheet (one entry per sheet):
column names, a semantic role for each column (date / metric / category / id / boolean / text / constant / empty),
statistics and a few sample rows.

Your jobs:
1. Understand what each sheet is about and how sheets relate.
2. Plan the most useful charts per sheet (max {max_charts} per sheet, fewer is fine). Prefer charts that answer
   real business questions: trend over time, top contributors, mix/share, comparison, distribution, relationship.
3. Detect the industry/business domain and market if the data makes it possible.

Chart rules (charts violating them are discarded):
- "type": one of line, area, bar, hbar, pie, scatter, histogram
- line/area: x MUST be a date column; set "time_grain" (day/week/month/quarter/year)
- bar/hbar/pie: x must be a category (or a date bucketed with time_grain). Use hbar for long labels / many items.
- pie only for share-of-total with <= 8 slices, never with group_by
- scatter: x and y must both be metrics
- histogram: x is a metric, no y
- "y" must be a metric, or omit y and use "agg":"count" to count rows
- "agg": sum, mean, median, count, min, max. Do NOT sum non-additive metrics like price, rating, stock level — use mean.
- "group_by" (optional): a category with few values, to split series
- NEVER use id, text, constant or empty columns in charts
- Never plan two charts that show the same thing

Return JSON:
{{
  "workbook_summary": "2-3 sentences: what this data is",
  "sheets": [
    {{"sheet": "<name>", "description": "1-2 sentences",
      "charts": [{{"type": "...", "x": "...", "y": "...", "agg": "...", "group_by": null,
                   "time_grain": null, "top_n": 10, "title": "Short human title",
                   "reason": "the business question this answers"}}]}}
  ],
  "industry": {{"name": "e.g. Fashion e-commerce", "sub_sector": "...", "market": "country/region or null",
               "confidence": 0.0-1.0, "evidence": "which columns/values suggest it",
               "own_brand": "company/brand name if visible in data, else null",
               "likely_competitors": ["only well-known real players in that market & sector"],
               "news_queries": ["3-4 short web-news search queries about this industry's latest trends in that market"]}}
}}"""


def ai_plan(profiles: list[dict], issues: list[dict], provider: str, context: dict) -> dict:
    payload = {
        "user_context": {k: v for k, v in context.items() if v},
        "data_quality_issues": [f"[{i['sheet']}] {i['title']}" for i in issues if i["severity"] != "low"][:20],
        "sheets": [compact_profile(p) for p in profiles if p["rows"]],
    }
    user = ("Here is the dataset profile. Plan the analysis.\n\n"
            + json.dumps(payload, ensure_ascii=False, default=str))
    return ask_json(PLAN_SYSTEM.format(max_charts=MAX_CHARTS_PER_SHEET), user, provider)


# --------------------------------------------------------------------------- #
# Fallback (no API key): a few sensible default charts
# --------------------------------------------------------------------------- #
def fallback_charts(profile: dict) -> list[dict]:
    br = profile["by_role"]
    metrics, dates = rank_metrics(profile), br.get("date", [])
    cats = sorted(br.get("category", []), key=lambda c: next(
        x["unique"] for x in profile["columns"] if x["name"] == c))
    cats = [c for c in cats if next(x["unique"] for x in profile["columns"] if x["name"] == c) > 1]
    charts: list[dict] = []
    m = metrics[0] if metrics else None
    agg = "sum" if m and is_additive(m) else "mean"

    if dates:
        charts.append({"type": "line", "x": dates[0], "y": m, "agg": agg if m else "count",
                       "title": f"{m or 'Rows'} over time", "reason": "How is it trending?"})
        if len(metrics) > 1:
            m2 = metrics[1]
            charts.append({"type": "line", "x": dates[0], "y": m2, "agg": "sum" if is_additive(m2) else "mean",
                           "title": f"{m2} over time", "reason": "Second key metric trend"})
    for c in cats[:3]:
        n_unique = next(x["unique"] for x in profile["columns"] if x["name"] == c)
        t = "pie" if n_unique <= 5 and agg == "sum" and not charts_have(charts, "pie") else \
            "hbar" if n_unique > 6 else "bar"
        charts.append({"type": t, "x": c, "y": m, "agg": agg if m else "count",
                       "title": f"{m or 'Rows'} by {c}",
                       "reason": f"Which {c} contributes most?" if agg == "sum" or not m else f"How does {m} compare across {c}?"})
    if m:
        charts.append({"type": "histogram", "x": m, "title": f"Distribution of {m}",
                       "reason": "Typical values and outliers"})
    if len(metrics) >= 2:
        charts.append({"type": "scatter", "x": metrics[0], "y": metrics[1],
                       "title": f"{metrics[0]} vs {metrics[1]}", "reason": "Are they related?"})
    return charts[:MAX_CHARTS_PER_SHEET]


def charts_have(charts, t):
    return any(c["type"] == t for c in charts)


INDUSTRY_KEYWORDS = {
    "Retail / E-commerce": r"order|sku|gmv|cart|shopee|tokopedia|lazada|tiktok shop|marketplace|product|checkout|basket",
    "Fashion & Apparel": r"dress|shirt|hijab|shoes|apparel|fashion|size|bag|jacket|celana|baju",
    "Food & Beverage": r"menu|food|drink|beverage|coffee|kopi|outlet|restaurant|meal|dine",
    "Finance / Banking": r"loan|credit|debit|account|balance|interest|saldo|transaction|bank|premium|claim",
    "Healthcare": r"patient|diagnos|doctor|hospital|clinic|medicine|obat|pasien",
    "Logistics": r"shipment|awb|resi|courier|kurir|warehouse|delivery|freight|weight",
    "Marketing / Advertising": r"impression|click|ctr|cpc|campaign|ad spend|roas|conversion|reach",
    "Education": r"student|course|grade|school|siswa|kelas|enrol",
    "Property / Real estate": r"property|rent|sewa|unit|tenant|sqm|lease",
    "Manufacturing": r"production|machine|defect|batch|yield|downtime|plant",
}


def fallback_industry(profiles: list[dict]) -> dict:
    words = []
    for p in profiles:
        for c in p["columns"]:
            words.append(c["name"])
            words += list((c.get("stats") or {}).get("top_values", {}).keys())
    blob = " ".join(map(str, words)).lower()
    scores = {k: len(re.findall(v, blob)) for k, v in INDUSTRY_KEYWORDS.items()}
    best = sorted(scores.items(), key=lambda kv: -kv[1])
    if not best or best[0][1] == 0:
        return {"name": None, "confidence": 0}
    names = [k for k, v in best[:2] if v]
    return {"name": " / ".join(names), "confidence": min(0.3 + 0.1 * best[0][1], 0.7),
            "evidence": "keyword match on column names and values (no AI)",
            "likely_competitors": [], "news_queries": [f"{names[0]} industry trends"]}


# --------------------------------------------------------------------------- #
def plan(profiles: list[dict], issues: list[dict], provider: str | None, context: dict) -> dict:
    """Return {"source", "workbook_summary", "industry", "sheets": {name: {...}}, "rejected": [...]}."""
    result = {"source": "fallback", "workbook_summary": None, "industry": None,
              "sheets": {}, "rejected": [], "error": None}
    raw = None
    if provider:
        try:
            raw = ai_plan(profiles, issues, provider, context)
            result["source"] = provider
            result["workbook_summary"] = raw.get("workbook_summary")
            result["industry"] = raw.get("industry")
        except AIError as e:
            result["error"] = str(e)

    ai_sheets = {s.get("sheet"): s for s in (raw or {}).get("sheets", [])}
    for p in profiles:
        if not p["rows"]:
            continue
        proposed = ai_sheets.get(p["sheet"], {}).get("charts") if raw else None
        used_fallback = not proposed
        proposed = proposed or fallback_charts(p)
        valid, seen = [], set()
        for spec in proposed:
            clean, why = validate_spec(spec, p)
            if not clean:
                result["rejected"].append({"sheet": p["sheet"], "spec": spec, "reason": why})
                continue
            key = (clean["type"], clean["x"], clean["y"], clean["group_by"])
            if key not in seen:
                seen.add(key)
                valid.append(clean)
        result["sheets"][p["sheet"]] = {
            "description": ai_sheets.get(p["sheet"], {}).get("description"),
            "charts": valid[:MAX_CHARTS_PER_SHEET],
            "used_fallback": used_fallback,
        }

    if not result["industry"] or not result["industry"].get("name"):
        result["industry"] = fallback_industry(profiles)
    # user-provided overrides always win
    if context.get("industry"):
        result["industry"]["name"] = context["industry"]
        result["industry"]["confidence"] = 1.0
    if context.get("competitors"):
        result["industry"]["likely_competitors"] = [c.strip() for c in context["competitors"].split(",") if c.strip()]
    if context.get("brand"):
        result["industry"]["own_brand"] = context["brand"]
    if context.get("market"):
        result["industry"]["market"] = context["market"]
    return result
