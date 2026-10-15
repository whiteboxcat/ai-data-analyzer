"""A small, safe query language the AI uses to get numbers from the cleaned data.

The AI never runs code. It sends a JSON description like

    {"sheet": "Sales",
     "filters": [{"column": "Channel", "op": "==", "value": "Shopee"},
                 {"column": "Order Date", "op": "between", "value": ["2025-03-01", "2025-03-31"]}],
     "group_by": ["City"],
     "time_grain": "month",                       # only used when grouping by a date
     "metrics": [{"column": "GMV", "agg": "sum"}, {"agg": "count"}],
     "sort": {"by": "sum of GMV", "desc": true},
     "limit": 10}

and Python validates every column and operator, then computes it with pandas.
Without "group_by" and "metrics" it returns matching rows (for example to look
at outliers). Results are capped so they stay small enough to send back.
"""
from __future__ import annotations

import pandas as pd

from .charts import PERIOD
from .utils import is_texty, parse_dates, parse_numeric_text, to_jsonable

OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "not in", "contains", "between", "is empty", "is not empty"}
AGGS = {"sum", "mean", "median", "min", "max", "count", "nunique"}
MAX_FILTERS, MAX_GROUP, MAX_METRICS, MAX_LIMIT = 8, 2, 5, 50


class QueryError(ValueError):
    """Something the AI can fix and retry (unknown column, bad operator...)."""


def _roles(profile: dict) -> dict[str, str]:
    return {c["name"]: c["role"] for c in profile["columns"]}


def _typed(s: pd.Series, role: str) -> pd.Series:
    if role == "date":
        return parse_dates(s)
    if role == "metric":
        return parse_numeric_text(s) if is_texty(s) else pd.to_numeric(s, errors="coerce")
    return s


def _col(name, roles: dict, what: str) -> str:
    if name in roles:
        return name
    # be forgiving about case and spaces
    for c in roles:
        if str(c).strip().lower() == str(name).strip().lower():
            return c
    raise QueryError(f"Unknown {what} column '{name}'. Available: {', '.join(roles)}")


def _value(v, role: str):
    if role == "date":
        d = parse_dates(pd.Series([v])).iloc[0]
        if pd.isna(d):
            raise QueryError(f"'{v}' is not a date (use YYYY-MM-DD)")
        return d
    if role == "metric":
        n = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.isna(n):
            raise QueryError(f"'{v}' is not a number")
        return n
    return str(v)


def _apply_filter(d: pd.DataFrame, f: dict, roles: dict) -> pd.Series:
    col = _col(f.get("column"), roles, "filter")
    op = str(f.get("op", "==")).strip().lower()
    if op not in OPS:
        raise QueryError(f"Unknown operator '{op}'. Use one of: {', '.join(sorted(OPS))}")
    role = roles[col]
    s = d[col]
    if op == "is empty":
        return s.isna() | (s.astype(str).str.strip() == "")
    if op == "is not empty":
        return s.notna() & (s.astype(str).str.strip() != "")
    raw = f.get("value")
    text_col = role not in ("date", "metric")
    if text_col:
        s = s.astype(str).str.strip()

    if op == "contains":
        return s.astype(str).str.contains(str(raw), case=False, regex=False, na=False)
    if op in ("in", "not in"):
        if not isinstance(raw, list):
            raw = [raw]
        vals = [_value(v, role) for v in raw]
        if text_col:  # case-insensitive match for labels
            mask = s.str.lower().isin([v.lower() for v in vals])
        else:
            mask = s.isin(vals)
        return ~mask if op == "not in" else mask
    if op == "between":
        if not isinstance(raw, list) or len(raw) != 2:
            raise QueryError("'between' needs a list of two values [from, to]")
        lo, hi = _value(raw[0], role), _value(raw[1], role)
        if role == "date" and isinstance(raw[1], str) and len(raw[1]) <= 10:
            hi = hi + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)  # include the whole end day
        return (s >= lo) & (s <= hi)

    v = _value(raw, role)
    if text_col and op in ("==", "!="):
        mask = s.str.lower() == v.lower()
        return ~mask if op == "!=" else mask
    if text_col:
        raise QueryError(f"'{op}' only works on numbers or dates; '{col}' is text")
    return {"==": s == v, "!=": s != v, ">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[op]


def run_query(sheets: dict[str, pd.DataFrame], profiles: dict[str, dict], q: dict) -> dict:
    """Execute one query. Raises QueryError with a message the AI can act on."""
    if not isinstance(q, dict):
        raise QueryError("The query must be a JSON object")
    name = q.get("sheet") or (next(iter(sheets)) if len(sheets) == 1 else None)
    if name not in sheets:
        raise QueryError(f"Unknown sheet '{name}'. Sheets: {', '.join(sheets)}")
    df, roles = sheets[name], _roles(profiles[name])

    filters = q.get("filters") or []
    group_by = q.get("group_by") or []
    metrics = q.get("metrics") or []
    if isinstance(group_by, str):
        group_by = [group_by]
    if len(filters) > MAX_FILTERS or len(group_by) > MAX_GROUP or len(metrics) > MAX_METRICS:
        raise QueryError(f"Too complex: at most {MAX_FILTERS} filters, {MAX_GROUP} group_by, {MAX_METRICS} metrics")
    limit = max(1, min(int(q.get("limit") or 20), MAX_LIMIT))

    group_cols = [_col(g, roles, "group_by") for g in group_by]
    metric_specs = []
    for m in metrics:
        agg = str(m.get("agg", "sum")).lower()
        if agg not in AGGS:
            raise QueryError(f"Unknown agg '{agg}'. Use one of: {', '.join(sorted(AGGS))}")
        col = m.get("column")
        if agg == "count" and not col:
            metric_specs.append((None, "count", "rows"))
            continue
        col = _col(col, roles, "metric")
        if agg not in ("count", "nunique") and roles[col] != "metric":
            raise QueryError(f"Can't {agg} '{col}': it isn't numeric ({roles[col]}). Use count or nunique.")
        metric_specs.append((col, agg, f"{agg} of {col}"))
    filter_cols = [_col(f.get("column"), roles, "filter") for f in filters]
    show_cols = [_col(c, roles, "columns") for c in (q.get("columns") or [])]

    # typed copy of just the columns we need
    used = list(dict.fromkeys(group_cols + [c for c, _, _ in metric_specs if c] + filter_cols + show_cols))
    if not used:
        used = list(df.columns)
    d = pd.DataFrame({c: _typed(df[c], roles[c]) for c in used}, index=df.index)

    mask = pd.Series(True, index=d.index)
    for f in filters:
        mask &= _apply_filter(d, f, roles).fillna(False)
    d = d[mask]
    matched = int(len(d))

    if group_cols or metric_specs:
        keys = []
        for c in group_cols:
            if roles[c] == "date":
                grain = q.get("time_grain") or "month"
                if grain not in PERIOD:
                    raise QueryError(f"time_grain must be one of: {', '.join(PERIOD)}")
                key = f"{c} ({grain})"
                d[key] = d[c].dt.to_period(PERIOD[grain]).dt.start_time.dt.strftime("%Y-%m-%d")
            else:
                key = c
                d[key] = d[c].astype(str).where(d[c].notna(), "(empty)")
            keys.append(key)
        if not metric_specs:
            metric_specs = [(None, "count", "rows")]

        def compute(frame) -> dict:
            out = {}
            for col, agg, label in metric_specs:
                if col is None:
                    out[label] = len(frame)
                elif agg == "count":
                    out[label] = int(frame[col].notna().sum())
                else:
                    out[label] = frame[col].agg(agg)
            return out

        if keys:
            res = pd.DataFrame([{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **compute(g)}
                                for k, g in d.groupby(keys, sort=False)])
        else:
            res = pd.DataFrame([compute(d)])
        groups_total = int(len(res))
        sort = q.get("sort") or {}
        by = sort.get("by") if isinstance(sort, dict) else None
        if by and by not in res.columns:
            raise QueryError(f"Can't sort by '{by}'. Result columns: {', '.join(res.columns)}")
        if not by:
            by = keys[0] if keys and any(roles[c] == "date" for c in group_cols) else res.columns[-1]
            desc = not (keys and any(roles[c] == "date" for c in group_cols))
        else:
            desc = bool(sort.get("desc", True))
        res = res.sort_values(by, ascending=not desc, kind="stable")
        truncated = groups_total > limit
        res = res.head(limit)
        for c in res.columns:
            if pd.api.types.is_float_dtype(res[c]):
                res[c] = res[c].round(2)
        kind, total = "aggregate", groups_total
    else:
        cols = show_cols or list(df.columns)[:12]
        res = df.loc[d.index, cols].head(limit)
        kind, total, truncated = "rows", matched, matched > limit

    return to_jsonable({
        "sheet": name, "kind": kind, "matched_rows": matched, "result_rows": total, "truncated": truncated,
        "columns": [str(c) for c in res.columns],
        "rows": res.astype(object).where(res.notna(), None).values.tolist(),
    })
