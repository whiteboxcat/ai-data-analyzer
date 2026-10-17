"""Chat with the AI about an analysed file.

Each question runs a short loop:
  1. The AI gets the dataset summary, the conversation so far and the question.
  2. It either answers, or asks for numbers with a query (see query.py).
  3. Python runs the query on the cleaned data and sends the result back.
  4. Repeat (at most MAX_STEPS), then the AI answers from the real numbers.
The AI never sees the full data and never runs code.
"""
from __future__ import annotations

import json
import pickle
import re
import threading
from pathlib import Path

from .ai_client import AIError, ask_json
from .planner import compact_profile
from .query import QueryError, run_query

MAX_STEPS = 4             # AI calls per question (queries + final answer)
MAX_HISTORY = 12          # previous messages sent along
MAX_QUESTION = 2000       # characters

SYSTEM = """You are a senior data analyst chatting with a business user about THEIR spreadsheet.
You get: a summary of every sheet (columns, their roles and stats, a few sample rows), the report's
key findings, the conversation so far, and the user's new question.

To get numbers you don't already have, ask for a query. Python runs it on the full cleaned data:
{{"action": "query", "why": "short reason", "query": {{
   "sheet": "<sheet>",
   "filters": [{{"column": "...", "op": "==|!=|>|>=|<|<=|in|not in|contains|between|is empty|is not empty", "value": ...}}],
   "group_by": ["<column>", ...],          // 0-2 columns; a date column is bucketed by time_grain
   "time_grain": "day|week|month|quarter|year",
   "metrics": [{{"column": "<numeric column>", "agg": "sum|mean|median|min|max|count|nunique"}}, {{"agg": "count"}}],
   "sort": {{"by": "<result column, e.g. 'sum of GMV'>", "desc": true}},
   "limit": 20 }}}}
- Without group_by and metrics the query returns matching rows (good for checking specific records).
- Dates as "YYYY-MM-DD". "between" takes [from, to] and includes both ends.
- Text matching is case-insensitive. Only use column names that exist.
- One query per step; you can ask again after seeing the result. If a query errors, fix it.

When you can answer:
{{"action": "answer", "answer": "your reply"}}
Rules for the answer:
- Use ONLY numbers from the summary or from query results. Never guess or invent figures.
  If the data can't answer the question, say so and suggest what data would.
- Be direct: lead with the answer, then 2-5 short supporting points. Plain text; you may use
  **bold** and lines starting with "- " for bullets. No tables, no headings.
- Format numbers with thousand separators. Mention filters/assumptions you used (e.g. date range).
- For questions about fraud, corruption or wrongdoing: describe patterns in the data as risk signals
  worth reviewing (e.g. missing vendors, round amounts, unusual concentration), say what further checks
  would confirm or rule them out, and never state or imply that a named person or company committed a crime.
- {language_rule}"""


def _from_text(text: str) -> dict | None:
    """Make sense of a reply that wasn't valid JSON.

    - Plain prose (the model ignored the format): use it as the answer.
    - JSON cut off mid-answer (hit the length limit): keep the answer text so far.
    - Anything else that looks like broken JSON: None, so the caller asks again.
    """
    t = text.strip()
    t = re.sub(r"^```(?:json|text|markdown)?\s*|\s*```$", "", t).strip()
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', t, re.S)
    if m:
        body = m.group(1).rstrip("\\")
        try:
            body = json.loads(f'"{body}"')
        except json.JSONDecodeError:
            body = body.replace("\\n", "\n").replace('\\"', '"')
        cut = not re.search(r'"answer"\s*:\s*"(?:[^"\\]|\\.)*"', t, re.S)
        return {"action": "answer", "answer": body.strip() + (" …\n\n(The reply was cut off. Ask a follow-up for the rest.)" if cut else "")}
    if t.startswith("{") or t.startswith("["):
        return None
    return {"action": "answer", "answer": t} if t else None


def build_context(result: dict) -> dict:
    """Compact summary of the report for the AI (no raw data)."""
    ins = result.get("insights") or {}
    return {
        "file": result.get("file"),
        "workbook_summary": result.get("workbook_summary"),
        "workbook_notes": result.get("workbook_notes"),
        "industry": (result.get("industry") or {}).get("name"),
        "sheets": [{**compact_profile(s["profile"]), "description": s.get("description"),
                    "kpis": s.get("kpis")} for s in result.get("sheets", [])],
        "report_findings": {
            "executive_summary": ins.get("executive_summary"),
            "insights": [f"{i.get('title')}: {i.get('detail')}" for i in (ins.get("insights") or [])][:8],
        },
        "data_quality_notes": [f"[{i['sheet']}] {i['title']}" for i in result.get("issues", [])][:15],
    }


# --------------------------------------------------------------------------- #
# Cleaned data stored next to each report
# --------------------------------------------------------------------------- #
_cache: dict[str, tuple] = {}
_lock = threading.Lock()


def save_data(path: Path, sheets: dict, profiles: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump({"sheets": sheets, "profiles": profiles}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(path: Path):
    """Our own pickle written by save_data for this report. Cached for the latest report only."""
    key = str(path)
    with _lock:
        if key in _cache:
            return _cache[key]
        if not path.exists():
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301 - file written by this app, never user-supplied
        _cache.clear()          # keep memory low on small servers: one report at a time
        _cache[key] = (data["sheets"], data["profiles"])
        return _cache[key]


# --------------------------------------------------------------------------- #
def ask(result: dict, data, history: list[dict], question: str, provider: str,
        language: str | None = None) -> dict:
    """Answer one question. Returns {"answer", "steps": [{why, query, result|error}], "provider"}."""
    question = (question or "").strip()[:MAX_QUESTION]
    if not question:
        raise ValueError("Type a question first.")
    rule = (f"Reply in {language}." if language
            else "Reply in the language the user writes in.")
    system = SYSTEM.format(language_rule=rule)
    context = build_context(result)
    convo = [{"role": m.get("role"), "text": str(m.get("text", ""))[:3000]}
             for m in (history or [])[-MAX_HISTORY:] if m.get("role") in ("user", "assistant")]
    steps: list[dict] = []

    for step in range(MAX_STEPS):
        last = step == MAX_STEPS - 1
        payload = {
            "dataset": context,
            "data_queries_available": data is not None,
            "conversation": convo,
            "question": question,
            "query_results_so_far": steps,
        }
        if last:
            payload["instruction"] = "No more queries allowed. Answer now with what you have."
        if data is None:
            payload["note"] = "Queries are unavailable for this report (re-run the analysis to enable them)."
        reply = ask_json(system, json.dumps(payload, ensure_ascii=False, default=str), provider,
                         max_tokens=8000, allow_text=True)
        if "_text" in reply:
            reply = _from_text(reply["_text"])
            if reply is None:           # cut-off or broken JSON: tell the AI and try again
                if last:
                    raise AIError("The AI's reply was cut off. Try asking again, or ask a narrower question.")
                steps.append({"why": "(unreadable reply)",
                              "error": "Your last reply was not valid JSON. Reply with exactly one JSON object."})
                continue
        action = reply.get("action")

        if action == "query" and not last:
            q = reply.get("query") or {}
            entry = {"why": reply.get("why"), "query": q}
            try:
                if data is None:
                    raise QueryError("Queries are unavailable for this report; answer from the summary.")
                entry["result"] = run_query(data[0], data[1], q)
            except QueryError as e:
                entry["error"] = str(e)
            except Exception as e:  # noqa: BLE001 - report to the AI, don't crash the chat
                entry["error"] = f"{e.__class__.__name__}: {e}"
            steps.append(entry)
            continue

        answer = reply.get("answer") or reply.get("text")
        if not answer:
            raise AIError("The AI replied without an answer. Try asking again.")
        return {"answer": str(answer), "steps": steps, "provider": provider}

    raise AIError("The AI couldn't finish the answer. Try a more specific question.")
