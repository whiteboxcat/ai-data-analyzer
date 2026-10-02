"""Data-quality checks and safe automatic fixes.

detect_issues()  -> list of issues (each with a recommendation)
apply_fixes()    -> cleaned DataFrame + a log of what changed

"Safe" fixes (auto_fix=True) never invent data: they remove exact duplicates,
trim spaces, unify spelling, convert types, and label missing categories as
"Unknown". Judgement calls (outliers, negative quantities, missing numbers)
are only *recommended* so the user decides.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .utils import is_texty, parse_dates, parse_numeric_text

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
NON_NEGATIVE_HINTS = re.compile(r"(qty|quantity|jumlah|price|harga|stock|stok|count|orders|visitors)", re.I)


def _issue(sheet, column, kind, severity, title, detail, recommendation,
           affected=0, auto_fix=False):
    return {
        "id": f"{sheet}|{column}|{kind}",
        "sheet": sheet, "column": column, "kind": kind,
        "severity": severity, "title": title, "detail": detail,
        "recommendation": recommendation, "affected_rows": int(affected),
        "auto_fix": auto_fix,
    }


def _case_variants(s: pd.Series) -> dict[str, str]:
    """Map spelling variants ("dress", "Dress ", "DRESS") to the most common form."""
    nn = s.dropna().astype(str)
    stripped = nn.str.strip().str.replace(r"\s+", " ", regex=True)
    key = stripped.str.casefold()
    mapping = {}
    for _, grp in stripped.groupby(key):
        if grp.nunique() > 1:
            canonical = grp.value_counts().idxmax()
            for v in grp.unique():
                if v != canonical:
                    mapping[v] = canonical
    return mapping


def detect_issues(df: pd.DataFrame, profile: dict) -> list[dict]:
    sheet = profile["sheet"]
    issues: list[dict] = []
    n = len(df)
    if n == 0:
        return issues

    dup = int(df.duplicated().sum())
    if dup:
        issues.append(_issue(sheet, None, "duplicate_rows", "high",
                             f"{dup} exact duplicate row{'s' if dup != 1 else ''}",
                             "Rows that are identical in every column usually come from double exports.",
                             "Remove duplicates, keeping the first occurrence.", dup, True))

    for col in profile["columns"]:
        name, role = col["name"], col["role"]
        s = df[name]

        if role == "empty":
            issues.append(_issue(sheet, name, "empty_column", "low", f"'{name}' is completely empty",
                                 "The column has no values.", "Drop the column.", n, True))
            continue
        if role == "constant":
            issues.append(_issue(sheet, name, "constant_column", "low", f"'{name}' has one single value",
                                 f"Every row is '{s.dropna().iloc[0]}'. It adds no analytical information.",
                                 "Keep it as context, but it will be ignored for charts."))

        # Whitespace / spelling variants --------------------------------------
        if is_texty(s) and role in ("category", "text", "id"):
            txt = s.dropna().astype(str)
            ws = int((txt != txt.str.strip().str.replace(r"\s+", " ", regex=True)).sum())
            variants = _case_variants(s) if role == "category" else {}
            if variants:
                examples = ", ".join(f"'{k}' → '{v}'" for k, v in list(variants.items())[:4])
                affected = int(s.astype(str).str.strip().isin(variants.keys()).sum() + ws)
                issues.append(_issue(sheet, name, "inconsistent_labels", "medium",
                                     f"Inconsistent spelling in '{name}'",
                                     f"Same label written differently: {examples}. "
                                     "This splits one group into several in charts.",
                                     "Trim spaces and unify to the most common spelling.", affected, True))
            elif ws:
                issues.append(_issue(sheet, name, "whitespace", "low", f"Extra spaces in '{name}'",
                                     f"{ws} values have leading/trailing or double spaces.",
                                     "Trim the spaces.", ws, True))

        # Wrong types ----------------------------------------------------------
        if role == "metric" and col.get("stored_as") == "text":
            parsed = parse_numeric_text(s)
            bad = int((s.notna() & parsed.isna()).sum())
            ex = s.dropna().astype(str).head(2).tolist()
            issues.append(_issue(sheet, name, "number_as_text", "high",
                                 f"'{name}' numbers are stored as text",
                                 f"Values like {ex} can't be summed or charted as numbers."
                                 + (f" {bad} values could not be read and will become blank." if bad else ""),
                                 "Convert to numeric (currency symbols and thousand separators removed).",
                                 int(s.notna().sum()), True))
        if role == "date" and col.get("stored_as"):
            parsed = parse_dates(s)
            bad = int((s.notna() & parsed.isna()).sum())
            fmt_note = ""
            if col["stored_as"] == "text":
                pats = s.dropna().astype(str).str.replace(r"\d", "9", regex=True).value_counts()
                if len(pats) > 1:
                    fmt_note = f" Mixed formats found: {', '.join(pats.index[:3])}."
            issues.append(_issue(sheet, name, "date_as_" + ("int" if "integer" in col["stored_as"] else "text"),
                                 "high" if fmt_note else "medium",
                                 f"'{name}' dates stored as {col['stored_as']}",
                                 "Dates must be real date values to build timelines." + fmt_note
                                 + (f" {bad} values could not be parsed." if bad else ""),
                                 "Convert to a proper date type.", int(s.notna().sum()), True))

        # Missing values -------------------------------------------------------
        miss, pct = col["missing"], col["missing_pct"]
        if miss and role not in ("empty",):
            sev = "high" if pct > 30 else "medium" if pct > 5 else "low"
            if role in ("category", "boolean"):
                rec, auto = "Fill with 'Unknown' so these rows still appear in group totals.", True
            elif role == "metric":
                rec, auto = ("Leave blank (ignored in sums/averages). If blank really means zero "
                             "(e.g. no discount), fill with 0 manually."), False
            elif role == "date":
                rec, auto = "Rows without a date are excluded from time charts. Check the source export.", False
            elif role == "id":
                rec, auto, sev = "Missing identifiers often mean broken or subtotal rows — review them.", False, "high"
            else:
                rec, auto = "Usually fine for free-text columns.", False
            if pct > 60:
                rec = f"{pct}% missing — consider dropping this column. " + rec
            issues.append(_issue(sheet, name, "missing_values", sev,
                                 f"{miss} missing values in '{name}' ({pct}%)",
                                 "Empty cells or blank text.", rec, miss, auto))

        # Duplicated IDs ---------------------------------------------------------
        if role == "id":
            d = int(s.dropna().duplicated().sum())
            if d and d != dup:
                issues.append(_issue(sheet, name, "duplicate_ids", "medium",
                                     f"{d} repeated values in ID column '{name}'",
                                     "Could be multiple line-items per order (fine) or double entries (not fine).",
                                     "If each row should be unique, investigate these rows.", d))

        # Suspicious numbers ----------------------------------------------------
        if role == "metric":
            v = parse_numeric_text(s) if is_texty(s) else s.astype(float)
            v = v.dropna()
            if len(v) >= 10:
                neg = int((v < 0).sum())
                if neg and NON_NEGATIVE_HINTS.search(name):
                    issues.append(_issue(sheet, name, "negative_values", "high",
                                         f"{neg} negative value{'s' if neg != 1 else ''} in '{name}'",
                                         f"'{name}' should not be negative. Could be returns or typos.",
                                         "Check these rows. If they are returns, move them to a separate column; "
                                         "otherwise correct or remove them.", neg))
                q1, q3 = v.quantile([.25, .75])
                iqr = q3 - q1
                if iqr > 0:
                    extreme = v[(v > q3 + 5 * iqr) | (v < q1 - 5 * iqr)]
                    if len(extreme):
                        issues.append(_issue(sheet, name, "extreme_outliers", "medium",
                                             f"{len(extreme)} extreme outlier{'s' if len(extreme) != 1 else ''} in '{name}'",
                                             f"Values far outside the normal range (typical {q1:,.0f}–{q3:,.0f}, "
                                             f"found up to {extreme.abs().max():,.0f}). "
                                             "One typo can distort totals and averages.",
                                             "Verify these rows. Fix typos; keep genuine big orders.", len(extreme)))

    issues.sort(key=lambda i: SEVERITY_ORDER[i["severity"]])
    return issues


def apply_fixes(df: pd.DataFrame, profile: dict, issues: list[dict],
                selected: set[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Apply auto-fixable issues (all of them, or only ids in `selected`)."""
    todo = [i for i in issues if i["auto_fix"] and (selected is None or i["id"] in selected)]
    by_kind: dict[str, list[dict]] = {}
    for i in todo:
        by_kind.setdefault(i["kind"], []).append(i)

    out = df.copy()
    log: list[str] = []

    # 1. blank strings -> missing
    for c in out.columns:
        if is_texty(out[c]):
            out[c] = out[c].where(~out[c].astype(str).str.strip().isin(["", "nan", "None"]), np.nan)

    # 2. drop empty columns
    for i in by_kind.get("empty_column", []):
        out = out.drop(columns=i["column"])
        log.append(f"Dropped empty column '{i['column']}'.")

    # 3. whitespace + spelling
    for i in by_kind.get("whitespace", []) + by_kind.get("inconsistent_labels", []):
        c = i["column"]
        before = out[c].copy()
        out[c] = out[c].where(out[c].isna(), out[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True))
        if i["kind"] == "inconsistent_labels":
            out[c] = out[c].replace(_case_variants(out[c]))
        changed = int((before.astype(str) != out[c].astype(str)).sum())
        log.append(f"Cleaned labels in '{c}' ({changed} values changed).")

    # 4. types
    for i in by_kind.get("number_as_text", []):
        c = i["column"]
        out[c] = parse_numeric_text(out[c])
        log.append(f"Converted '{c}' from text to numbers.")
    for i in by_kind.get("date_as_text", []) + by_kind.get("date_as_int", []):
        c = i["column"]
        out[c] = parse_dates(out[c])
        log.append(f"Converted '{c}' to real dates ({int(out[c].isna().sum())} blank after conversion).")

    # 5. duplicates (after normalising, so near-identical rows are caught too)
    if by_kind.get("duplicate_rows"):
        before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {before - len(out)} duplicate rows.")

    # 6. missing categories
    for i in by_kind.get("missing_values", []):
        c = i["column"]
        if c in out.columns:
            k = int(out[c].isna().sum())
            out[c] = out[c].astype(object).where(out[c].notna(), "Unknown")
            log.append(f"Filled {k} missing values in '{c}' with 'Unknown'.")

    return out.infer_objects(), log
