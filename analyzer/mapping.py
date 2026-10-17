"""Use lookup ("master") sheets to standardise names across a workbook.

Business workbooks often include a table like

    product_master:  sku_raw    | sku_canonical | ...
                     FBL 200ml  | FBL200
                     FBL-200    | FBL200

Other sheets then use the messy spellings (sku_raw). This module finds such
lookup tables and adds the standard column (sku_canonical) next to the messy
one in every other sheet, so totals by product or distributor add up. The
original column is kept.

A lookup is a pair of columns in one sheet: "<thing>_raw" (or "<thing>",
"<thing>_name", ...) and "<thing>_canonical" (or _clean, _std, _standard,
_master, _normalized, _mapped).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

_RAW_SUFFIX = re.compile(r"[_\s-]*(raw|name|original|input|src|source|text)$", re.I)
_CANON_SUFFIX = re.compile(r"^[_\s-]*(canonical|clean|cleaned|std|standard|standardi[sz]ed|master|"
                           r"normali[sz]ed|mapped|official|code)$", re.I)


def _norm(v) -> str | None:
    """Matching key: case, extra spaces and surrounding punctuation don't matter."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    t = re.sub(r"\s+", " ", str(v)).strip().lower()
    return t or None


@dataclass
class Lookup:
    sheet: str
    raw_col: str
    canon_col: str
    stem: str
    table: dict[str, str] = field(default_factory=dict)       # norm(raw) -> canonical
    conflicts: set[str] = field(default_factory=set)            # raw values mapped to 2+ canonicals


def find_lookups(sheets: dict[str, pd.DataFrame]) -> list[Lookup]:
    found = []
    for name, df in sheets.items():
        cols = [str(c) for c in df.columns]
        for a in cols:
            stem = _RAW_SUFFIX.sub("", a).strip().lower()
            if not stem:
                continue
            for b in cols:
                if b == a or not b.lower().startswith(stem):
                    continue
                if not _CANON_SUFFIX.match(b[len(stem):]):
                    continue
                pairs = df[[a, b]].dropna()
                if len(pairs) < 2 or pairs[a].nunique() < 0.8 * len(pairs):
                    continue          # a lookup lists each raw spelling about once
                lk = Lookup(sheet=name, raw_col=a, canon_col=b, stem=stem)
                seen: dict[str, set] = {}
                for raw, canon in pairs.itertuples(index=False):
                    key = _norm(raw)
                    if key:
                        seen.setdefault(key, set()).add(str(canon).strip())
                    ckey = _norm(canon)
                    if ckey:                          # data already using the standard name
                        seen.setdefault(ckey, set()).add(str(canon).strip())
                for key, canons in seen.items():
                    if len(canons) == 1:
                        lk.table[key] = next(iter(canons))
                    else:
                        lk.conflicts.add(key)
                if lk.table:
                    found.append(lk)
    return found


def _target_columns(df: pd.DataFrame, lk: Lookup) -> list[str]:
    """Columns in a data sheet that hold the same thing as the lookup's raw column."""
    out = []
    for c in map(str, df.columns):
        cl = c.lower()
        if cl == lk.canon_col.lower():
            continue
        if cl == lk.raw_col.lower() or _RAW_SUFFIX.sub("", c).strip().lower() == lk.stem:
            vals = {_norm(v) for v in df[c].dropna().unique()} - {None}
            if vals and len(vals & lk.table.keys()) >= 0.5 * len(vals):
                out.append(c)
    return out


def _issue(sheet, column, kind, severity, title, detail, recommendation, affected, auto_fix, **extra):
    return {"id": f"{sheet}|{column}|{kind}", "sheet": sheet, "column": column, "kind": kind,
            "severity": severity, "title": title, "detail": detail, "recommendation": recommendation,
            "affected_rows": int(affected), "auto_fix": auto_fix, **extra}


def standardise(sheets: dict[str, pd.DataFrame], apply: bool = True) -> tuple[dict, dict, dict]:
    """Find lookups and use them on the other sheets.

    Returns (issues by sheet, cleaning-log lines by sheet, reference sheets
    {lookup sheet: [sheets it standardised]}). With apply=True the standard
    column is inserted next to the messy one (sheets are changed in place).
    """
    issues: dict[str, list] = {}
    logs: dict[str, list] = {}
    refs: dict[str, list] = {}
    for lk in find_lookups(sheets):
        for name, df in sheets.items():
            if name == lk.sheet or lk.canon_col in map(str, df.columns):
                continue
            for col in _target_columns(df, lk):
                keys = df[col].map(_norm)
                mapped = keys.map(lambda k: lk.table.get(k) if k else None)
                present = df[col].notna()
                unmatched = df.loc[present & mapped.isna(), col].astype(str).str.strip()

                # spellings that end up as the same standard name
                pairs = pd.DataFrame({"raw": df[col].astype(str).str.strip(), "canon": mapped})[present & mapped.notna()]
                groups = pairs.groupby("canon")["raw"].agg(lambda s: sorted(set(s)))
                merged = {c: v for c, v in groups.items() if len(v) > 1}
                renamed = {c: v[0] for c, v in groups.items() if len(v) == 1 and v[0] != c}

                parts = [", ".join(f"'{x}'" for x in v) + f" → {c}" for c, v in list(merged.items())[:4]]
                detail = (f"Matched against '{lk.raw_col}' in the {lk.sheet} sheet; the standard name goes in a "
                          f"new '{lk.canon_col}' column next to '{col}'. ")
                if parts:
                    detail += "Merged spellings: " + "; ".join(parts) + ". "
                if renamed and not merged:
                    detail += f"{len(renamed)} name(s) renamed to their standard form. "
                if len(unmatched):
                    counts = unmatched.value_counts()
                    shown = ", ".join(f"'{v}' ({n} row{'s' if n != 1 else ''})" for v, n in counts.head(4).items())
                    detail += f"Not found in {lk.sheet}: {shown}; kept as written in '{lk.canon_col}'."
                n_merged_rows = int(pairs["raw"].isin([x for v in merged.values() for x in v]).sum())
                title = (f"'{col}' has {sum(len(v) for v in merged.values())} spellings for "
                         f"{len(merged)} item{'s' if len(merged) != 1 else ''}: standardised with {lk.sheet}"
                         if merged else f"'{col}' standardised with {lk.sheet}")
                issues.setdefault(name, []).append(_issue(
                    name, col, "mapped_names", "high" if merged else "low", title, detail.strip(),
                    f"Use '{lk.canon_col}' for totals and comparisons. The original '{col}' column is kept."
                    + (f" Add the missing values to {lk.sheet} if they should be included." if len(unmatched) else ""),
                    n_merged_rows or int(present.sum()), True, unmatched=int(len(unmatched))))
                refs.setdefault(lk.sheet, [])
                if name not in refs[lk.sheet]:
                    refs[lk.sheet].append(name)
                if apply:
                    pos = list(df.columns).index(col) + 1
                    # values not in the lookup keep their own (trimmed) text, e.g. 'ALL'
                    fallback = df[col].astype(str).str.strip().where(present, None)
                    df.insert(pos, lk.canon_col, mapped.where(mapped.notna(), fallback).astype(object))
                    df.attrs.setdefault("mapped_cols", []).append(col)
                    logs.setdefault(name, []).append(
                        f"Added '{lk.canon_col}' from {lk.sheet} ({int(mapped.notna().sum())} rows matched"
                        + (f", {len(unmatched)} not found" if len(unmatched) else "") + ").")
    return issues, logs, refs
