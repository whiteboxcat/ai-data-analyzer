"""Apply the user's own cleaning instructions before the automatic cleaning.

The user types plain language ("drop the RERUN rows", "keep the latest
inventory snapshot per distributor and SKU", "don't fill blanks with Unknown").
The AI turns it into operations from a small fixed list, Python checks every
column and value, then applies them. Anything the AI can't express is reported
as "not possible" instead of being guessed. The automatic cleaning then handles
the rest, minus any automatic fix the user switched off.
"""
from __future__ import annotations

import json

import pandas as pd

from .ai_client import AIError, ask_json
from .planner import compact_profile
from .profiler import profile_sheet
from .query import QueryError, _apply_filter, _col, _typed
from .utils import parse_dates, parse_numeric_text

MAX_TEXT = 3000
MAX_OPS = 30

# automatic fixes the user can switch off
AUTO_KINDS = {
    "duplicate_rows": "remove exact duplicate rows",
    "reloaded_rows": "remove rows loaded twice (e.g. BASE vs RERUN)",
    "inconsistent_labels": "unify spelling/case of labels",
    "whitespace": "trim extra spaces",
    "number_as_text": "convert numbers stored as text",
    "date_as_text": "convert dates stored as text",
    "date_as_int": "convert YYYYMMDD numbers to dates",
    "missing_values": "fill missing categories with 'Unknown'",
    "empty_column": "drop empty columns",
    "mapped_names": "standardise names with lookup sheets",
}

SYSTEM = """You turn a user's data-cleaning instructions into operations a program will run.
You get the instructions and a profile of every sheet (columns, roles, stats, sample rows).

Allowed operations (JSON objects):
- {{"op": "drop_rows", "sheet": "...", "filters": [FILTER, ...], "description": "..."}}   remove matching rows
- {{"op": "keep_rows", "sheet": "...", "filters": [FILTER, ...], "description": "..."}}   keep only matching rows
- {{"op": "drop_columns", "sheet": "...", "columns": ["..."], "description": "..."}}
- {{"op": "rename_values", "sheet": "...", "column": "...", "mapping": {{"old": "new"}}, "description": "..."}}
- {{"op": "fill_missing", "sheet": "...", "column": "...", "value": ..., "description": "..."}}
- {{"op": "dedupe", "sheet": "...", "keys": ["col", ...], "keep": "first|last", "order_by": "<column or null>",
    "description": "..."}}   one row per key combination; with order_by, rows are sorted by it first,
    so keep "last" = the latest/highest
- {{"op": "set_type", "sheet": "...", "column": "...", "type": "date|number|text", "description": "..."}}
- {{"op": "skip_auto_fix", "kinds": ["..."], "sheet": "<sheet or null for all>", "column": "<column or null>",
    "description": "..."}}   switch off automatic fixes (for one column when the user names one). Kinds: {kinds}
FILTER = {{"column": "...", "op": "==|!=|>|>=|<|<=|in|not in|contains|between|is empty|is not empty", "value": ...}}
  Text matching is case-insensitive. Dates as "YYYY-MM-DD".

Rules:
- Only do what the instructions ask. Don't add your own cleaning: the program cleans the rest itself.
- Use only sheet and column names from the profile. If an instruction names something loosely
  ("the sales sheet", "SKU"), pick the clear match; if it's ambiguous, put it in not_possible.
- An instruction that applies to "all sheets" becomes one operation per sheet that has the column.
- "description": one short plain sentence of what the step does, in the user's language.
- If something can't be expressed with these operations, add it to not_possible with a short reason.

Return JSON: {{"operations": [...], "not_possible": [{{"instruction": "...", "reason": "..."}}], "summary": "one sentence"}}"""


class InstructionError(ValueError):
    pass


def _roles(profile: dict) -> dict[str, str]:
    return {c["name"]: c["role"] for c in profile["columns"]}


def _sheet(op: dict, sheets: dict) -> str:
    name = op.get("sheet")
    if name in sheets:
        return name
    for n in sheets:                              # forgiving about case
        if str(n).lower() == str(name).lower():
            return n
    raise InstructionError(f"There's no sheet called '{name}'.")


def _replace(sheets, name, new: pd.DataFrame, old: pd.DataFrame):
    new.attrs = dict(old.attrs)                   # keep notes like ignored side tables
    sheets[name] = new.reset_index(drop=True)


def _filter_role(s: pd.Series, role: str, filters: list[dict]) -> str:
    """Small sheets get profiled as categories; compare numbers and dates as such anyway."""
    if role in ("date", "metric"):
        return role
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return "metric"
    if any(str(f.get("op", "")).strip() in (">", ">=", "<", "<=", "between") for f in filters):
        present = s.dropna()
        if len(present) and parse_numeric_text(present.astype(str)).notna().mean() > 0.9:
            return "metric"
        if len(present) and parse_dates(present.astype(str)).notna().mean() > 0.9:
            return "date"
    return role


def _describe_filters(filters) -> str:
    parts = []
    for f in filters:
        v = f.get("value")
        v = ", ".join(map(str, v)) if isinstance(v, list) else v
        parts.append(f"{f.get('column')} {f.get('op', '==')} {v}" if v is not None else f"{f.get('column')} {f.get('op')}")
    return " and ".join(parts)


def _apply(op: dict, sheets: dict, profiles: dict, skip: dict) -> str:
    kind = op.get("op")
    if kind == "skip_auto_fix":
        kinds = [k for k in op.get("kinds") or [] if k in AUTO_KINDS]
        if not kinds:
            raise InstructionError("No known automatic fix to switch off.")
        target = op.get("sheet") or "*"
        if target != "*":
            target = _sheet(op, sheets)
        col = op.get("column")
        if col:
            if target == "*":
                raise InstructionError("Name the sheet when switching a fix off for one column.")
            col = _col(col, _roles(profiles[target]), "column")
        skip.setdefault(target, set()).update(f"{k}@{col}" if col else k for k in kinds)
        where = "all sheets" if target == "*" else f"{target}.{col}" if col else target
        return f"Switched off in {where}: " + "; ".join(AUTO_KINDS[k] for k in kinds) + "."

    name = _sheet(op, sheets)
    df, roles = sheets[name], _roles(profiles[name])

    if kind in ("drop_rows", "keep_rows"):
        filters = op.get("filters") or []
        if not filters:
            raise InstructionError("No condition given for which rows to remove or keep.")
        cols = [_col(f.get("column"), roles, "filter") for f in filters]
        roles = {**roles, **{c: _filter_role(df[c], roles[c], [f for f in filters if _col(f.get("column"), roles, "filter") == c])
                             for c in cols}}
        typed = pd.DataFrame({c: _typed(df[c], roles[c]) for c in dict.fromkeys(cols)}, index=df.index)
        mask = pd.Series(True, index=df.index)
        for f in filters:
            mask &= _apply_filter(typed, f, roles).fillna(False)
        n = int(mask.sum())
        cond = _describe_filters(filters)
        if kind == "drop_rows":
            if n == len(df):
                raise InstructionError(f"That would remove every row of {name} ({cond}); not applied.")
            _replace(sheets, name, df[~mask], df)
            return f"Removed {n:,} row{'s' if n != 1 else ''} from {name} where {cond}."
        if n == 0:
            raise InstructionError(f"No rows in {name} match {cond}; nothing kept, so not applied.")
        _replace(sheets, name, df[mask], df)
        return f"Kept {n:,} of {len(df):,} rows in {name} where {cond}."

    if kind == "drop_columns":
        cols = [_col(c, roles, "column") for c in op.get("columns") or []]
        if not cols:
            raise InstructionError("No columns given.")
        _replace(sheets, name, df.drop(columns=cols), df)
        return f"Dropped {', '.join(cols)} from {name}."

    if kind == "rename_values":
        col = _col(op.get("column"), roles, "column")
        mapping = op.get("mapping") or {}
        if not isinstance(mapping, dict) or not mapping:
            raise InstructionError("No values to rename.")
        keyed = {str(k).strip().lower(): v for k, v in mapping.items()}
        text = df[col].astype(str).str.strip().str.lower()
        hit = text.isin(keyed.keys()) & df[col].notna()
        new = df.copy()
        new.loc[hit, col] = text[hit].map(keyed)
        _replace(sheets, name, new, df)
        pairs = ", ".join(f"'{k}' → '{v}'" for k, v in list(mapping.items())[:5])
        return f"Renamed {int(hit.sum()):,} value{'s' if hit.sum() != 1 else ''} in {name}.{col}: {pairs}."

    if kind == "fill_missing":
        col = _col(op.get("column"), roles, "column")
        blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        new = df.copy()
        value = op.get("value")
        if roles[col] == "metric":
            value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(value):
                raise InstructionError(f"'{op.get('value')}' isn't a number, and {col} is numeric.")
        new[col] = new[col].astype(object).where(~blank, value)
        _replace(sheets, name, new.infer_objects(), df)
        return f"Filled {int(blank.sum()):,} blank cell{'s' if blank.sum() != 1 else ''} in {name}.{col} with {value!r}."

    if kind == "dedupe":
        keys = [_col(k, roles, "key") for k in op.get("keys") or []]
        if not keys:
            raise InstructionError("No key columns given for removing duplicates.")
        keep = "last" if str(op.get("keep", "first")).lower() == "last" else "first"
        order = op.get("order_by")
        work = df
        if order:
            order = _col(order, roles, "order_by")
            sort_key = _typed(df[order], roles[order])
            if roles[order] not in ("date", "metric"):        # e.g. ISO timestamps stored as text
                as_date = parse_dates(df[order])
                sort_key = as_date if as_date.notna().mean() > 0.8 else df[order].astype(str)
            work = df.assign(_order=sort_key).sort_values("_order", kind="stable").drop(columns="_order")
        out = work[~work.duplicated(subset=keys, keep=keep)].sort_index()
        n = len(df) - len(out)
        _replace(sheets, name, out, df)
        how = f" (kept the {'latest' if keep == 'last' else 'earliest'} by {order})" if order else f" (kept the {keep})"
        return f"Removed {n:,} duplicate row{'s' if n != 1 else ''} in {name} by {', '.join(keys)}{how}."

    if kind == "set_type":
        col = _col(op.get("column"), roles, "column")
        typ = str(op.get("type", "")).lower()
        new = df.copy()
        if typ == "date":
            new[col] = parse_dates(df[col])
        elif typ == "number":
            new[col] = parse_numeric_text(df[col])
        elif typ == "text":
            new[col] = df[col].astype(str).where(df[col].notna(), None)
        else:
            raise InstructionError(f"Unknown type '{typ}'. Use date, number or text.")
        bad = int((df[col].notna() & new[col].isna()).sum())
        _replace(sheets, name, new, df)
        return f"Set {name}.{col} to {typ}" + (f" ({bad} values couldn't be converted and are now blank)." if bad else ".")

    raise InstructionError(f"Unknown operation '{kind}'.")


def run(sheets: dict[str, pd.DataFrame], text: str, provider: str | None) -> dict | None:
    """Apply instructions to `sheets` in place. Returns the report, or None if there were none."""
    text = (text or "").strip()[:MAX_TEXT]
    if not text:
        return None
    report = {"instructions": text, "steps": [], "skip": {}, "summary": None, "error": None}
    if not provider:
        report["error"] = ("Cleaning instructions need an AI model to read them. Choose one on the upload page; "
                           "the automatic cleaning was applied without them.")
        return report

    profiles = {n: profile_sheet(df, n) for n, df in sheets.items()}
    payload = {"instructions": text, "sheets": [compact_profile(p) for p in profiles.values()]}
    system = SYSTEM.format(kinds=", ".join(f"{k} ({v})" for k, v in AUTO_KINDS.items()))
    try:
        plan = ask_json(system, json.dumps(payload, ensure_ascii=False, default=str), provider, max_tokens=6000)
    except AIError as e:
        report["error"] = f"The AI couldn't read the instructions, so they weren't applied. {e}"
        return report

    skip: dict[str, set] = {}
    for op in (plan.get("operations") or [])[:MAX_OPS]:
        if not isinstance(op, dict):
            continue
        step = {"description": op.get("description") or op.get("op"), "op": op}
        try:
            step["detail"] = _apply(op, sheets, profiles, skip)
            step["status"] = "done"
            if op.get("sheet") in sheets:
                profiles[op["sheet"]] = profile_sheet(sheets[op["sheet"]], op["sheet"])
        except (InstructionError, QueryError) as e:
            step["status"], step["detail"] = "failed", str(e)
        except Exception as e:  # noqa: BLE001 - one bad step must not stop the others
            step["status"], step["detail"] = "failed", f"{e.__class__.__name__}: {e}"
        report["steps"].append(step)
    for item in plan.get("not_possible") or []:
        if isinstance(item, dict):
            report["steps"].append({"description": item.get("instruction"), "status": "not_possible",
                                    "detail": item.get("reason")})
        elif item:
            report["steps"].append({"description": str(item), "status": "not_possible", "detail": ""})
    report["summary"] = plan.get("summary")
    report["skip"] = {k: sorted(v) for k, v in skip.items()}
    return report


def is_skipped(report: dict | None, sheet: str, kind: str, column=None) -> bool:
    """Did the user switch this automatic fix off (everywhere, for this sheet, or this column)?"""
    if not report:
        return False
    s = report.get("skip") or {}
    off = set(s.get("*", [])) | set(s.get(sheet, []))
    return kind in off or (column is not None and f"{kind}@{column}" in off)
