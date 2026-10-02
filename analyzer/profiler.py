"""Dataset profiler.

For each column we decide a *semantic role*, which is what the AI and the
chart engine care about (not the raw pandas dtype):

    date      - timeline column (real dates, date strings, 20240101 ints)
    metric    - numeric value you can SUM/AVG (GMV, Qty, Discount ...)
    id        - identifier, never chart or sum it (Order No, SKU, Invoice)
    category  - low-cardinality label to group by (Category, City, Channel)
    boolean   - yes/no flags
    text      - free text / high-cardinality strings (names, notes)
    empty     - no data at all
    constant  - one single value everywhere
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .utils import (date_parse_rate, is_texty, numeric_text_rate, parse_dates,
                    parse_numeric_text, to_jsonable)

# Name hints in English + Bahasa Indonesia.
ID_HINTS = re.compile(
    r"(^id$|_id$|^id_|\bid\b|\bno\b|\bno\.|number|nomor|\bnum\b|code|kode|sku|"
    r"invoice|faktur|order.?no|order.?id|resi|awb|uuid|key|ref)", re.I)
DATE_HINTS = re.compile(
    r"(date|time|tanggal|tgl|waktu|period|periode|month|bulan|year|tahun|day|hari|created|updated)", re.I)
METRIC_HINTS = re.compile(
    r"(amount|price|harga|total|qty|quantity|jumlah|sales|penjualan|revenue|pendapatan|gmv|"
    r"net|gross|cost|biaya|profit|laba|discount|diskon|fee|tax|pajak|margin|value|nilai|"
    r"spend|budget|stock|stok|count|score|rate|rating|weight|berat|visitors|orders|clicks)", re.I)

FREE_TEXT_HINTS = re.compile(
    r"(note|comment|remark|catatan|keterangan|description|deskripsi|desc$|address|alamat|message|pesan|review|ulasan|feedback)", re.I)

CATEGORY_MAX_UNIQUE = 50
SAMPLE_ROWS = 8


def _infer_role(name: str, s: pd.Series) -> tuple[str, dict]:
    """Return (role, extra_info) for one column."""
    nn = s.dropna()
    n, nunique = len(nn), nn.nunique()
    info: dict = {}

    if n == 0:
        return "empty", info
    if nunique == 1:
        return "constant", info

    uniq_ratio = nunique / n

    # Boolean ---------------------------------------------------------------
    if pd.api.types.is_bool_dtype(s):
        return "boolean", info
    if nunique == 2:
        vals = {str(v).strip().lower() for v in nn.unique()}
        if vals <= {"yes", "no", "y", "n", "true", "false", "1", "0", "ya", "tidak", "t", "f"}:
            return "boolean", info

    # Date ------------------------------------------------------------------
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date", info
    if pd.api.types.is_numeric_dtype(s):
        if date_parse_rate(s) > 0.95:  # e.g. 20240131
            info["stored_as"] = "integer YYYYMMDD"
            return "date", info
    elif is_texty(s):
        rate = date_parse_rate(s)
        if rate > 0.8 or (rate > 0.5 and DATE_HINTS.search(name)):
            info["stored_as"] = "text"
            info["parse_rate"] = round(rate, 3)
            return "date", info

    # ID --------------------------------------------------------------------
    name_says_id = bool(ID_HINTS.search(name)) and not METRIC_HINTS.search(name)
    if name_says_id and uniq_ratio > 0.5:
        return "id", info
    if is_texty(s) and uniq_ratio > 0.95 and n > 20:
        sample = nn.astype(str).head(200)
        # Codes like INV-00123, A12B9: no spaces, contain digits.
        if (sample.str.contains(r"\d") & ~sample.str.contains(r"\s")).mean() > 0.8:
            return "id", info
    if pd.api.types.is_integer_dtype(s) and uniq_ratio > 0.98 and n > 20:
        # A strictly increasing run of integers is almost always a row id.
        if nn.is_monotonic_increasing and not METRIC_HINTS.search(name):
            return "id", info

    # Metric ----------------------------------------------------------------
    if pd.api.types.is_numeric_dtype(s):
        if nunique <= 7 and not METRIC_HINTS.search(name) and ((nn % 1) == 0).all():
            info["note"] = "few distinct integers - could be a rating or code"
            return "category", info
        return "metric", info
    if is_texty(s) and numeric_text_rate(s) > 0.9:
        info["stored_as"] = "text"  # cleaner will offer to convert
        return "metric", info

    # Category vs free text ---------------------------------------------------
    if FREE_TEXT_HINTS.search(name):
        return "text", info
    if (nunique <= CATEGORY_MAX_UNIQUE and (uniq_ratio < 0.9 or n < 20)) or uniq_ratio < 0.3:
        return "category", info
    return "text", info


def _column_stats(role: str, s: pd.Series) -> dict:
    nn = s.dropna()
    if role == "metric":
        v = parse_numeric_text(nn) if is_texty(nn) else nn.astype(float)
        v = v.dropna()
        if v.empty:
            return {}
        q1, q3 = v.quantile([0.25, 0.75])
        return {
            "min": v.min(), "max": v.max(), "mean": v.mean(), "median": v.median(),
            "std": v.std(), "sum": v.sum(),
            "zeros": int((v == 0).sum()), "negatives": int((v < 0).sum()),
            "outliers_iqr": int(((v < q1 - 1.5 * (q3 - q1)) | (v > q3 + 1.5 * (q3 - q1))).sum()),
        }
    if role == "date":
        d = parse_dates(nn).dropna()
        if d.empty:
            return {}
        span = (d.max() - d.min()).days
        gaps = d.drop_duplicates().sort_values().diff().dt.days.dropna()
        step = gaps.median() if len(gaps) else None
        grain = ("day" if step is not None and step <= 1 else
                 "week" if step is not None and step <= 7 else
                 "month" if step is not None and step <= 31 else
                 "quarter" if step is not None and step <= 92 else "year")
        return {"min": d.min(), "max": d.max(), "span_days": span, "grain": grain,
                "suggested_chart_grain": "day" if span <= 45 else "week" if span <= 200
                else "month" if span <= 1100 else "year"}
    if role in ("category", "boolean"):
        vc = nn.astype(str).value_counts()
        return {"top_values": vc.head(10).to_dict(), "other_values": int(vc.iloc[10:].sum())}
    if role in ("text", "id"):
        lens = nn.astype(str).str.len()
        return {"example_values": nn.astype(str).head(3).tolist(),
                "avg_length": lens.mean()}
    return {}


def profile_sheet(df: pd.DataFrame, sheet_name: str = "Sheet1") -> dict:
    """Build a JSON-safe profile of one DataFrame."""
    columns = []
    for col in df.columns:
        s = df[col]
        role, info = _infer_role(str(col), s)
        missing = int(s.isna().sum() + (s.astype(str).str.strip() == "").sum()
                      if is_texty(s) else s.isna().sum())
        columns.append({
            "name": str(col),
            "dtype": str(s.dtype),
            "role": role,
            **info,
            "missing": missing,
            "missing_pct": round(missing / len(df) * 100, 2) if len(df) else 0,
            "unique": int(s.nunique(dropna=True)),
            "stats": _column_stats(role, s),
        })

    roles = {c["name"]: c["role"] for c in columns}
    sample = pd.concat([df.head(SAMPLE_ROWS // 2),
                        df.sample(min(SAMPLE_ROWS // 2, len(df)), random_state=0)]
                       ).drop_duplicates() if len(df) else df

    return to_jsonable({
        "sheet": sheet_name,
        "rows": len(df),
        "columns_count": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
        "by_role": {r: [c for c, rr in roles.items() if rr == r]
                    for r in ["date", "metric", "category", "id", "boolean", "text", "empty", "constant"]},
        "sample_rows": sample.astype(object).where(sample.notna(), None).to_dict(orient="records"),
    })


def profile_workbook(sheets: dict[str, pd.DataFrame]) -> dict:
    return {"sheets": [profile_sheet(df, name) for name, df in sheets.items()]}
