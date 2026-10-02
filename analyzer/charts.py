"""Validate chart specs (from AI or fallback) and execute them with pandas.

A chart spec looks like:
    {"type": "line", "x": "Order Date", "y": "GMV", "agg": "sum",
     "time_grain": "month", "group_by": null, "top_n": 10,
     "title": "...", "reason": "..."}

Python owns the maths: we aggregate the data ourselves and hand the browser a
small figure description (no raw rows) that static/charts.js draws as SVG, plus the aggregated table so the AI can
write insights from real numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import is_texty, parse_dates, parse_numeric_text, to_jsonable

CHART_TYPES = {"line", "area", "bar", "hbar", "pie", "scatter", "histogram"}
AGGS = {"sum", "mean", "median", "count", "min", "max", "nunique"}
PERIOD = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
GRAINS = PERIOD
MAX_SERIES = 8      # categorical palette has 8 slots; the rest folds into "Other"
MAX_BARS = 15


def _roles(profile: dict) -> dict[str, str]:
    return {c["name"]: c["role"] for c in profile["columns"]}


def validate_spec(spec: dict, profile: dict) -> tuple[dict | None, str | None]:
    """Return (clean_spec, None) or (None, reason_rejected)."""
    roles = _roles(profile)
    t = str(spec.get("type", "")).lower()
    x, y, g = spec.get("x"), spec.get("y"), spec.get("group_by")
    agg = str(spec.get("agg") or ("count" if not y else "sum")).lower()

    if t not in CHART_TYPES:
        return None, f"unknown chart type '{t}'"
    for col in (x, y, g):
        if col and col not in roles:
            return None, f"column '{col}' does not exist"
    if agg not in AGGS:
        return None, f"unknown aggregation '{agg}'"
    if any(roles.get(c) in ("id", "empty", "constant", "text") for c in (x, g) if c):
        return None, "cannot group by an ID / text / empty column"
    if y and roles.get(y) != "metric" and agg not in ("count", "nunique"):
        return None, f"'{y}' is not a numeric metric, can't {agg} it"
    if g and roles.get(g) not in ("category", "boolean"):
        return None, "group_by must be a category"

    if t in ("line", "area"):
        if roles.get(x) != "date":
            return None, "line/area charts need a date on the x-axis"
    elif t in ("bar", "hbar", "pie"):
        if roles.get(x) not in ("category", "boolean", "date"):
            return None, f"{t} needs a category on x"
        if t == "pie" and g:
            return None, "pie charts can't be grouped"
    elif t == "scatter":
        if roles.get(x) != "metric" or roles.get(y) != "metric":
            return None, "scatter needs two numeric metrics"
    elif t == "histogram":
        if roles.get(x) != "metric":
            return None, "histogram needs a numeric metric"

    clean = {
        "type": t, "x": x, "y": y, "group_by": g, "agg": agg,
        "time_grain": spec.get("time_grain") if spec.get("time_grain") in GRAINS else None,
        "top_n": max(3, min(int(spec.get("top_n") or 10), MAX_BARS)),
        "title": spec.get("title") or f"{agg.title()} of {y or 'rows'} by {x}",
        "reason": spec.get("reason", ""),
    }
    return clean, None


# --------------------------------------------------------------------------- #
def _prep(df: pd.DataFrame, cols: list[str], roles: dict) -> pd.DataFrame:
    """Copy the needed columns with proper types (in case data wasn't cleaned)."""
    out = pd.DataFrame(index=df.index)
    for c in dict.fromkeys(c for c in cols if c):
        s = df[c]
        if roles[c] == "date":
            out[c] = parse_dates(s)
        elif roles[c] == "metric":
            out[c] = parse_numeric_text(s) if is_texty(s) else pd.to_numeric(s, errors="coerce")
        else:
            out[c] = s.astype(str).str.strip().where(s.notna(), "Unknown")
    return out


def _agg(frame, by, y, agg):
    if agg == "count" or not y:
        return frame.groupby(by, observed=True).size().rename("value")
    return frame.groupby(by, observed=True)[y].agg(agg).rename("value")


def _fold_other(series: pd.Series, keep: int) -> list:
    """Top `keep` labels by frequency; everything else becomes 'Other'."""
    return series.value_counts().head(keep).index.tolist()


def build_chart(df: pd.DataFrame, spec: dict, profile: dict) -> dict:
    """Aggregate the data and return a small, library-free figure:

    figure = {"kind": line|area|bar|hbar|pie|scatter|histogram,
              "x_label", "y_label",
              "categories": [...],                      # x values (line/bar/hist bins)
              "series": [{"name", "slot", "values": [...]}]   # or "points": [[x, y], ...]
             }
    "slot" is the fixed palette position (0-7) so colour follows the entity.
    """
    roles = _roles(profile)
    t, x, y, g, agg = spec["type"], spec["x"], spec["y"], spec["group_by"], spec["agg"]
    d = _prep(df, [x, y, g], roles)
    if x:
        d = d[d[x].notna()]
    if y and agg != "count":
        d = d[d[y].notna()]

    y_label = f"{agg} of {y}" if y and agg != "count" else "Rows"
    fig = {"kind": t, "x_label": x, "y_label": y_label, "categories": [], "series": []}
    table: list[dict] = []
    extra: dict = {}

    # Fold group_by into top N + "Other" (fixed palette, never cycled).
    # Scatter overlaps marks, so it gets fewer distinguishable colours.
    if g:
        keep = _fold_other(d[g], (3 if t == "scatter" else MAX_SERIES) - 1)
        d[g] = d[g].where(d[g].isin(keep), "Other")

    def group_order():
        totals = d.groupby(g)[y].sum() if y and agg != "count" else d[g].value_counts()
        order = totals.sort_values(ascending=False).index.tolist()
        if "Other" in order:  # Other always last
            order.remove("Other")
            order.append("Other")
        return order

    if t in ("line", "area"):
        grain = spec["time_grain"] or profile_grain(profile, x)
        d["_period"] = d[x].dt.to_period(PERIOD[grain]).dt.start_time
        periods = sorted(d["_period"].unique())
        fig["categories"] = [pd.Timestamp(p).date().isoformat() for p in periods]
        groups = [(None, d)] if not g else [(k, d[d[g] == k]) for k in group_order()]
        for i, (name, part) in enumerate(groups):
            s = _agg(part, "_period", y, agg).reindex(periods)
            if agg in ("sum", "count"):
                s = s.fillna(0)
            fig["series"].append({"name": str(name) if name is not None else y_label, "slot": i,
                                  "values": s.round(2).tolist()})
            table += [{"period": pd.Timestamp(p).date().isoformat(), **({"group": name} if g else {}), "value": v}
                      for p, v in s.items()]
        fig["x_label"] = f"{x} ({grain})"
        # A partial last bucket (data ends 3 Dec) makes the line look like a crash.
        first, last = d[x].min(), d[x].max()
        partial = bool(grain != "day" and pd.notna(last)
                       and last.normalize() < last.to_period(PERIOD[grain]).end_time.normalize())
        partial_first = bool(grain != "day" and pd.notna(first)
                             and first.normalize() > first.to_period(PERIOD[grain]).start_time.normalize())
        extra["partial_last_period"] = fig["partial_last"] = partial
        extra["partial_first_period"] = fig["partial_first"] = partial_first

    elif t in ("bar", "hbar"):
        is_time = roles[x] == "date"
        if is_time:  # bar over time -> bucket first
            grain = spec["time_grain"] or profile_grain(profile, x)
            d[x] = d[x].dt.to_period(PERIOD[grain]).astype(str)
        if g:
            p = (d.groupby([x, g], observed=True)[y].agg(agg) if y and agg != "count"
                 else d.groupby([x, g], observed=True).size()).unstack(g).fillna(0)
            p = p.sort_index() if is_time else p.loc[p.sum(axis=1).sort_values(ascending=False).index]
            p = p.iloc[:spec["top_n"]] if not is_time else p
            p = p[[c for c in group_order() if c in p.columns]]
            fig["categories"] = p.index.astype(str).tolist()
            for i, col in enumerate(p.columns):
                fig["series"].append({"name": str(col), "slot": i, "values": p[col].round(2).tolist()})
            table = [{"x": idx, "group": c, "value": v} for idx, row in p.iterrows() for c, v in row.items()]
        else:
            s = _agg(d, x, y, agg)
            s = s.sort_index() if is_time else s.sort_values(ascending=False)
            if not is_time and len(s) > spec["top_n"]:
                other = s.iloc[spec["top_n"]:]
                s = s.iloc[:spec["top_n"]]
                if agg in ("sum", "count"):
                    s = pd.concat([s, pd.Series({"Other": other.sum()})])
            fig["categories"] = s.index.astype(str).tolist()
            fig["series"].append({"name": y_label, "slot": 0, "values": s.round(2).tolist()})
            table = [{"x": k, "value": v} for k, v in zip(fig["categories"], fig["series"][0]["values"])]

    elif t == "pie":
        s = _agg(d, x, y, agg).sort_values(ascending=False)
        if len(s) > MAX_SERIES:
            s = pd.concat([s.iloc[:MAX_SERIES - 1], pd.Series({"Other": s.iloc[MAX_SERIES - 1:].sum()})])
        fig["categories"] = s.index.astype(str).tolist()
        fig["series"].append({"name": y_label, "slot": 0, "values": s.round(2).tolist()})
        table = [{"x": k, "value": v} for k, v in s.items()]

    elif t == "scatter":
        sample = d.sample(min(len(d), 1500), random_state=0)
        groups = [(None, sample)] if not g else [(k, sample[sample[g] == k]) for k in group_order()]
        for i, (name, part) in enumerate(groups):
            fig["series"].append({"name": str(name) if name is not None else f"{y} vs {x}", "slot": i,
                                  "points": part[[x, y]].round(2).values.tolist()})
        fig["y_label"] = y
        corr = d[[x, y]].corr().iloc[0, 1]
        table = [{"correlation": round(float(corr), 3) if pd.notna(corr) else None, "points": len(d)}]

    elif t == "histogram":
        v = d[x].dropna()
        q1, q3 = v.quantile([.25, .75])  # keep one typo from squashing every bar
        lo, hi = max(v.min(), q1 - 5 * (q3 - q1)), min(v.max(), q3 + 5 * (q3 - q1))
        inside = v[(v >= lo) & (v <= hi)] if hi > lo else v
        counts, edges = np.histogram(inside, bins=min(20, max(5, inside.nunique())))
        fig["categories"] = [f"{edges[i]:,.0f}–{edges[i + 1]:,.0f}" for i in range(len(counts))]
        fig["series"].append({"name": "Rows", "slot": 0, "values": counts.tolist()})
        fig["y_label"] = "Rows"
        fig["note"] = (f"{len(v) - len(inside)} extreme value(s) outside {lo:,.0f}–{hi:,.0f} not shown"
                       if len(v) != len(inside) else None)
        q = v.quantile([0, .25, .5, .75, 1]).round(2)
        table = [{"min": q.iloc[0], "p25": q.iloc[1], "median": q.iloc[2], "p75": q.iloc[3], "max": q.iloc[4]}]

    return to_jsonable({**spec, **extra, "figure": fig, "table": table[:60]})


def profile_grain(profile: dict, col: str) -> str:
    for c in profile["columns"]:
        if c["name"] == col:
            return c.get("stats", {}).get("suggested_chart_grain", "month")
    return "month"


def kpis(df: pd.DataFrame, profile: dict, limit: int = 4) -> list[dict]:
    """Headline numbers: row count, date range, totals of main metrics."""
    roles = _roles(profile)
    out = [{"label": "Rows", "value": len(df), "format": "int"}]
    dates = [c for c, r in roles.items() if r == "date"]
    if dates:
        s = profile_col(profile, dates[0]).get("stats", {})
        if s.get("min"):
            out.append({"label": f"{dates[0]} range", "value": s["min"][:10], "format": "text",
                        "sub": f"to {s['max'][:10]} ({s['span_days']} days)"})
    from .planner import rank_metrics  # avoid circular import at module load
    from .planner import is_additive
    for m in rank_metrics(profile)[:limit]:
        st = profile_col(profile, m).get("stats", {})
        if "sum" not in st:
            continue
        if is_additive(m):
            out.append({"label": f"Total {m}", "value": st["sum"], "format": "num",
                        "sub": f"avg {_n(st['mean'])} per row"})
        else:
            out.append({"label": f"Average {m}", "value": st["mean"], "format": "num",
                        "sub": f"range {_n(st['min'])} – {_n(st['max'])}"})
    return to_jsonable(out)


def _n(v) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:,.2f}"


def profile_col(profile: dict, name: str) -> dict:
    return next((c for c in profile["columns"] if c["name"] == name), {})
