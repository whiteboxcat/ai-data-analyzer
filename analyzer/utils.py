"""Shared helpers: JSON-safe conversion, smart date and number parsing."""
from __future__ import annotations

import math
import re
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# JSON safety
# --------------------------------------------------------------------------- #
def to_jsonable(obj):
    """Recursively convert numpy / pandas / datetime values into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def is_texty(s: pd.Series) -> bool:
    """True for object / string columns (pandas 2 and 3 compatible)."""
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def _looks_dayfirst(values: pd.Series) -> bool:
    """dd/mm/yyyy if any first number is > 12 (common in Indonesia/Europe)."""
    first = values.astype(str).str.extract(r"^\s*(\d{1,2})[/\-.]\d{1,2}[/\-.]\d{2,4}")[0]
    first = pd.to_numeric(first, errors="coerce").dropna()
    return bool(len(first) and (first > 12).any())


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse a column into datetimes. Handles strings, Excel dates and YYYYMMDD ints."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    if pd.api.types.is_numeric_dtype(s):
        # Only treat integers shaped like 20240131 as dates.
        nn = s.dropna()
        if len(nn) and ((nn % 1) == 0).all() and nn.between(19000101, 21001231).all():
            return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        return pd.Series(pd.NaT, index=s.index)

    text = s.astype(str).str.strip().where(s.notna())
    # ISO dates (2025-03-12) are always year-month-day; parse them separately so a
    # dayfirst=True guess for "12/03/2025" values never flips them.
    iso = text.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", na=False)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if iso.any():
            out[iso] = pd.to_datetime(text[iso], errors="coerce", format="mixed", dayfirst=False)
        rest = ~iso & text.notna()
        if rest.any():
            out[rest] = pd.to_datetime(text[rest], errors="coerce", format="mixed",
                                       dayfirst=_looks_dayfirst(text[rest]))
    return out


def date_parse_rate(s: pd.Series) -> float:
    nn = s.dropna()
    if nn.empty:
        return 0.0
    # Plain small numbers / short codes are not dates.
    if is_texty(nn):
        sample = nn.astype(str).head(200)
        if not sample.str.contains(r"\d", regex=True).mean() > 0.8:
            return 0.0
        if sample.str.fullmatch(r"\d{1,6}").mean() > 0.5:
            return 0.0
    parsed = parse_dates(nn.head(500))
    return float(parsed.notna().mean())


# --------------------------------------------------------------------------- #
# Numbers stored as text: "Rp 1.250.000", "1,200.50", "(300)", "15%"
# --------------------------------------------------------------------------- #
_CURRENCY = re.compile(r"(rp\.?|idr|usd|us\$|\$|€|£|¥|sgd|myr|eur)", re.I)


def _dot_is_thousands(values: pd.Series) -> bool:
    v = values.astype(str).str.replace(_CURRENCY, "", regex=True).str.strip()
    thousands = v.str.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?").sum()
    decimal = v.str.fullmatch(r"-?\d+\.\d{1,2}").sum() + v.str.fullmatch(r"-?\d{1,3}(,\d{3})+\.\d+").sum()
    return thousands > decimal


def parse_numeric_text(s: pd.Series) -> pd.Series:
    """Convert messy numeric text to floats. Unparseable values become NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    text = s.astype(str).where(s.notna())
    dot_thousands = _dot_is_thousands(text.dropna())

    def conv(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return np.nan
        t = str(x).strip()
        if not t or t.lower() in {"nan", "none", "null", "-", "n/a", "na"}:
            return np.nan
        neg = t.startswith("(") and t.endswith(")")
        pct = t.endswith("%")
        t = _CURRENCY.sub("", t).replace("%", "").replace("(", "").replace(")", "")
        t = t.replace(" ", "").replace(" ", "")
        if dot_thousands:
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
        try:
            val = float(t)
        except ValueError:
            return np.nan
        if neg:
            val = -val
        return val / 100 if pct else val

    return text.map(conv).astype(float)


def numeric_text_rate(s: pd.Series) -> float:
    """Share of non-null text values that can be read as numbers."""
    nn = s.dropna()
    if nn.empty or not is_texty(nn):
        return 0.0
    return float(parse_numeric_text(nn.head(1000)).notna().mean())
