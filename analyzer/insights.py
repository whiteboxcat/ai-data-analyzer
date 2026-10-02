"""Insights + recommendations.

The AI gets the *aggregated results* of the charts Python actually built, so
every insight is grounded in real numbers instead of guesses.
"""
from __future__ import annotations

import json

from .ai_client import AIError, ask_json

INSIGHT_SYSTEM = """You are a senior business/data analyst writing for a busy manager.
You receive: the dataset description, headline KPIs, the aggregated numbers behind each chart, and the
data-quality findings. Write insights ONLY from these numbers — never invent figures. Quote concrete numbers
(with % changes / shares where you can compute them). Be specific and practical, no generic advice.
{language_rule}
Return JSON:
{{
  "executive_summary": "3-5 sentences, the most important story in the data",
  "chart_insights": {{"<chart_id>": "1-2 sentence takeaway for that chart"}},
  "insights": [{{"title": "short", "detail": "what the numbers show and why it matters", "sheet": "<sheet>"}}],
  "recommendations": [{{"title": "action", "detail": "what to do and expected impact",
                        "priority": "high|medium|low", "based_on": "which finding"}}],
  "data_caveats": ["limitations the reader should know (data quality, short time range, ...)"],
  "next_questions": ["good follow-up questions to explore with more data"]
}}
Give 4-8 insights and 3-6 recommendations."""


def ai_insights(bundle: dict, provider: str, language: str | None = None) -> dict:
    rule = (f"Write all text in {language}." if language
            else "Write in the same language as the user's context if given, otherwise English.")
    user = "Analysis results:\n\n" + json.dumps(bundle, ensure_ascii=False, default=str)
    return ask_json(INSIGHT_SYSTEM.format(language_rule=rule), user, provider, max_tokens=10000)


def build_bundle(plan: dict, sheet_results: list[dict], issues: list[dict], context: dict) -> dict:
    """Everything the insight writer needs, compact."""
    return {
        "user_context": {k: v for k, v in context.items() if v},
        "workbook_summary": plan.get("workbook_summary"),
        "industry": (plan.get("industry") or {}).get("name"),
        "sheets": [{
            "sheet": s["sheet"], "description": s.get("description"), "rows": s["rows"],
            "kpis": s["kpis"],
            "charts": [{"chart_id": c["id"], "title": c["title"], "type": c["type"], "x": c["x"], "y": c["y"],
                        "agg": c["agg"], "group_by": c["group_by"], "data": c["table"][:40]}
                       for c in s["charts"]],
        } for s in sheet_results],
        "data_quality": [f"[{i['sheet']}] {i['title']}" for i in issues][:25],
    }


# --------------------------------------------------------------------------- #
# Fallback without AI: simple, honest, computed observations
# --------------------------------------------------------------------------- #
def fallback_insights(sheet_results: list[dict], issues: list[dict]) -> dict:
    insights, chart_notes = [], {}
    for s in sheet_results:
        for c in s["charts"]:
            t, tab = c["type"], c["table"]
            if not tab:
                continue
            if t in ("line", "area") and not c["group_by"] and len(tab) >= 3:
                # compare complete periods only; partial first/last buckets mislead
                full = tab[1 if c.get("partial_first_period") else 0: -1 if c.get("partial_last_period") else None]
                if len(full) >= 2:
                    tab = full
                first, last = tab[0]["value"], tab[-1]["value"]
                peak = max(tab, key=lambda r: r["value"] or 0)
                change = (last - first) / first * 100 if first else None
                note = (f"From {tab[0]['period']} to {tab[-1]['period']}: {first:,.0f} → {last:,.0f}"
                        + (f" ({change:+.1f}%)." if change is not None else ".")
                        + f" Peak {peak['value']:,.0f} at {peak['period']}."
                        + (" (Incomplete first/last periods excluded.)"
                           if c.get("partial_last_period") or c.get("partial_first_period") else ""))
                chart_notes[c["id"]] = note
                insights.append({"title": c["title"], "detail": note, "sheet": s["sheet"]})
            elif t in ("bar", "hbar", "pie") and not c["group_by"] and c["agg"] not in ("sum", "count"):
                top = max(tab, key=lambda r: r["value"] or 0)
                low = min(tab, key=lambda r: r["value"] or 0)
                chart_notes[c["id"]] = (f"Highest {c['agg']}: '{top['x']}' ({top['value']:,.2f}); "
                                        f"lowest: '{low['x']}' ({low['value']:,.2f}).")
            elif t in ("bar", "hbar", "pie") and not c["group_by"]:
                total = sum(r["value"] or 0 for r in tab)
                top = max(tab, key=lambda r: r["value"] or 0)
                if total:
                    note = f"'{top['x']}' is the largest at {top['value']:,.0f} ({top['value'] / total:.0%} of the total)."
                    chart_notes[c["id"]] = note
                    insights.append({"title": c["title"], "detail": note, "sheet": s["sheet"]})
            elif t == "scatter" and tab[0].get("correlation") is not None:
                r = tab[0]["correlation"]
                strength = "strong" if abs(r) > .7 else "moderate" if abs(r) > .4 else "weak"
                chart_notes[c["id"]] = f"Correlation {r:+.2f} ({strength})."
            elif t == "histogram":
                q = tab[0]
                chart_notes[c["id"]] = f"Half of rows fall between {q['p25']:,.0f} and {q['p75']:,.0f} (median {q['median']:,.0f})."
    high = [i for i in issues if i["severity"] == "high" and not i["auto_fix"]]
    return {
        "executive_summary": None,
        "chart_insights": chart_notes,
        "insights": insights[:8],
        "recommendations": [{"title": f"Review: {i['title']}", "detail": i["recommendation"],
                             "priority": "high", "based_on": "data quality"} for i in high[:4]],
        "data_caveats": ["These observations are computed without AI. Add an API key in .env "
                         "to get written insights and business recommendations."],
        "next_questions": [],
    }


def generate(plan, sheet_results, issues, context, provider) -> dict:
    if provider:
        try:
            out = ai_insights(build_bundle(plan, sheet_results, issues, context), provider, context.get("language"))
            out["source"] = provider
            return out
        except AIError as e:
            fb = fallback_insights(sheet_results, issues)
            fb["source"], fb["error"] = "fallback", str(e)
            return fb
    fb = fallback_insights(sheet_results, issues)
    fb["source"] = "fallback"
    return fb
