"""Load CSV / XLSX files into a dict of {sheet_name: DataFrame}.

Every sheet of an Excel workbook is read. Sheets often have a title or blank
rows above the real header, so we detect the header row instead of assuming
it is row 0.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

MAX_HEADER_SCAN = 15   # rows to scan when looking for the header
HEAD_BYTES = 256 * 1024  # sample used to sniff encoding, delimiter and header


def _detect_header_row(raw: pd.DataFrame) -> int:
    """Return the index of the row that most looks like a header.

    A header row is wide (many non-empty cells) and mostly text.
    """
    if raw.empty:
        return 0
    width = raw.notna().sum(axis=1).max()
    best_row, best_score = 0, -1.0
    for i in range(min(MAX_HEADER_SCAN, len(raw))):
        row = raw.iloc[i]
        filled = row.notna().sum()
        if filled < max(1, width * 0.6):
            continue
        text_share = sum(isinstance(v, str) for v in row.dropna()) / filled
        unique_share = row.dropna().astype(str).nunique() / filled
        score = text_share + unique_share
        if score > best_score:
            best_row, best_score = i, score
        if text_share == 1 and unique_share == 1:
            break  # perfect header, stop early
    return best_row


def _finalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Promote the detected header row, drop fully-empty rows/columns."""
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return pd.DataFrame()
    raw = raw.reset_index(drop=True)
    h = _detect_header_row(raw)
    header = raw.iloc[h].tolist()

    cols, seen = [], {}
    for i, c in enumerate(header):
        name = str(c).strip() if pd.notna(c) and str(c).strip() else f"column_{i + 1}"
        if name in seen:  # de-duplicate header names
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)

    df = raw.iloc[h + 1:].copy()
    df.columns = cols
    df = df.reset_index(drop=True)
    # Columns were read as mixed objects (header row was part of the data).
    # Convert a column to numeric only if EVERY non-empty value is a clean number;
    # messy columns ("1,200", "Rp 5.000") stay as text for the cleaner to flag.
    for c in df.columns:
        s = df[c]
        if s.dtype == object:
            non_null = s.dropna()
            if non_null.empty:
                continue
            conv = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")
            if conv.notna().all():
                df[c] = pd.to_numeric(s.astype(str).str.strip().replace({"nan": None, "": None}),
                                      errors="coerce")
    return df.infer_objects()


def _sniff_sep(text: str) -> str:
    """Pick , ; tab or | — whichever splits the first lines most consistently."""
    lines = [l for l in text.splitlines()[:30] if l.strip()]
    best, best_score = "\x1f", -1  # \x1f never appears: one single column
    for sep in [",", ";", "\t", "|"]:
        with_sep = [l for l in lines if sep in l]  # title lines have none
        if not with_sep:
            continue
        # "1.200,50" is a decimal comma, not a delimiter
        if sep == "," and all(re.fullmatch(r'\s*"?\(?-?[\d.]*,\d+\)?"?\s*', l) for l in with_sep):
            continue
        counts = [l.count(sep) for l in with_sep]
        # consistent column count across lines beats raw frequency
        common = max(set(counts), key=counts.count)
        score = counts.count(common) * 10 + common
        if score > best_score:
            best, best_score = sep, score
    return best


def _detect_encoding(head: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            # ignore a multi-byte character cut in half at the end of the sample
            head.decode(enc) if len(head) < HEAD_BYTES else head[:-4].decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _clean_columns(cols) -> list[str]:
    """Blank headers -> column_N; pandas' duplicate names ('GMV.1') -> 'GMV_1'."""
    out, seen = [], set()
    for i, c in enumerate(cols):
        name = "" if c is None or (isinstance(c, float) and pd.isna(c)) else str(c).strip()
        if not name or name.startswith("Unnamed:"):
            name = f"column_{i + 1}"
        m = re.fullmatch(r"(.+)\.(\d+)", name)
        if m and m.group(1) in seen:
            name = f"{m.group(1)}_{m.group(2)}"
        base, n = name, 1
        while name in seen:
            n += 1
            name = f"{base}_{n}"
        seen.add(name)
        out.append(name)
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    """Two passes: find the header on a small sample, then read the whole file
    with pandas' fast C parser so numeric columns are stored compactly
    (float64 = 8 bytes per cell instead of ~50+ for a Python string)."""
    with open(path, "rb") as f:
        head = f.read(HEAD_BYTES)
    enc = _detect_encoding(head)
    text = head.decode(enc, errors="ignore")
    sep = _sniff_sep(text)

    # fixed width so short title lines and wide data rows both fit
    width = max((line.count(sep) for line in text.splitlines()[:MAX_HEADER_SCAN + 5]), default=0) + 1
    sample = pd.read_csv(io.StringIO(text), header=None, names=range(width), sep=sep, engine="python",
                         dtype=object, skip_blank_lines=True, nrows=MAX_HEADER_SCAN + 5, on_bad_lines="skip")
    sample = sample.dropna(how="all").dropna(axis=1, how="all")
    if sample.empty:
        return pd.DataFrame()
    header_label = sample.index[_detect_header_row(sample.reset_index(drop=True))]

    df = pd.read_csv(path, header=header_label, sep=sep, encoding=enc, encoding_errors="replace",
                     engine="c" if len(sep) == 1 else "python", skip_blank_lines=True,
                     low_memory=False, on_bad_lines="skip", skipinitialspace=False)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    df.columns = _clean_columns(df.columns)
    return df


def _excel_engine() -> str | None:
    """calamine (Rust) reads .xlsx several times faster with less memory, if installed."""
    try:
        import python_calamine  # noqa: F401
        return "calamine"
    except ImportError:
        return None


def _col_letter(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (Excel column letters)."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(65 + r) + out
    return out


_PIVOT_WORDS = re.compile(r"^(row labels|column labels|grand total|(sum|count|average|max|min) of .+)$", re.I)


def _split_blocks(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Separate the main table from small tables placed beside it.

    People often put a pivot table or a summary to the right of their data,
    after an empty column. Read as one grid, that summary turns into fake
    columns full of blanks. Here the grid is cut at fully empty columns; the
    biggest block that isn't an Excel pivot is the data, and pivots or smaller
    blocks are set aside and reported.
    """
    filled = raw.notna().any(axis=0).tolist()
    blocks, start = [], None
    for i, f in enumerate(filled + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            blocks.append((start, i - 1))
            start = None
    if len(blocks) <= 1:
        return raw, []
    sizes = [int(raw.iloc[:, a:b + 1].notna().sum().sum()) for a, b in blocks]
    parts = [raw.iloc[:, a:b + 1].dropna(how="all") for a, b in blocks]
    texts = [{str(v).strip() for v in p.values.ravel() if isinstance(v, str)} for p in parts]
    # Excel pivot tables give themselves away with their own labels.
    pivot = [any(_PIVOT_WORDS.search(t) for t in tx) for tx in texts]
    candidates = [k for k in range(len(blocks)) if not pivot[k]] or list(range(len(blocks)))
    main = max(candidates, key=lambda k: sizes[k])
    keep, side = [main], []
    main_part = parts[main]
    main_header = {str(v).strip() for v in main_part.iloc[0].dropna().tolist()}
    for k, (a, b) in enumerate(blocks):
        if k == main:
            continue
        part = parts[k]
        looks_pivot = pivot[k]
        # a pivot filter cell repeats a column name of the main table ("extract_version | BASE")
        repeats_header = bool(texts[k] & main_header)
        # Otherwise: a big block starting on the same row is part of the table
        # (a table with one empty column inside it), not a side table.
        same_top = part.index.min() == main_part.index.min()
        if not looks_pivot and not repeats_header and sizes[k] >= 0.5 * sizes[main] and same_top:
            keep.append(k)
            continue
        first = [str(v) for v in part.iloc[0].dropna().tolist()][:3]
        side.append({"columns": f"{_col_letter(a)}–{_col_letter(b)}" if b > a else _col_letter(a),
                     "cells": sizes[k], "starts_with": first})
    cols = [c for k in sorted(keep) for c in range(blocks[k][0], blocks[k][1] + 1)]
    return raw.iloc[:, cols], side


_DOC_NAME = re.compile(r"read\s*_?me|^notes?$|^about|instruction|dictionary|glossary|definition|"
                       r"metadata|^legend$|^info$|^keterangan|^catatan", re.I)


def _doc_text(raw: pd.DataFrame, name: str) -> str | None:
    """Text of a notes / data-dictionary sheet, or None if it's a data sheet.

    These sheets explain the data ("duplicate reruns may exist") instead of
    holding it. They're given to the AI as context and not charted.
    """
    grid = raw.dropna(how="all").dropna(axis=1, how="all")
    if grid.empty:
        return None
    cells = [v for v in grid.values.ravel() if v is not None and not (isinstance(v, float) and pd.isna(v))]
    texts = [str(v).strip() for v in cells if isinstance(v, str) and str(v).strip()]
    all_text = len(texts) == len(cells)
    wordy = bool(texts) and sum(len(t) for t in texts) / len(texts) >= 25
    if not (_DOC_NAME.search(name) and all_text) and not (all_text and wordy and len(grid) <= 40):
        return None
    lines = [" | ".join(str(v).strip() for v in row if v is not None and not (isinstance(v, float) and pd.isna(v)))
             for row in grid.itertuples(index=False)]
    return "\n".join(line for line in lines if line)[:6000]


def frames_to_sheets(raw_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Raw grids (no header yet), e.g. from the Google Sheets API, to clean sheets.
    Notes sheets come back as a one-column frame with attrs["doc_text"]."""
    out = {}
    for name, raw in raw_frames.items():
        raw = raw.reset_index(drop=True)
        raw.columns = range(raw.shape[1])
        text = _doc_text(raw, str(name))
        if text:
            doc = pd.DataFrame({"note": text.split("\n")})
            doc.attrs["doc_text"] = text
            out[str(name)] = doc
            continue
        main, side = _split_blocks(raw)
        df = _finalize(main)
        if not df.empty:
            if side:
                df.attrs["side_tables"] = side
            out[str(name)] = df
    return out


def load_file(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load a CSV or Excel file. Returns {sheet_name: DataFrame}."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        df = _read_csv(path)
        return {"CSV": df} if not df.empty else {}

    if ext in (".xlsx", ".xlsm", ".xls"):
        engine = _excel_engine() if ext != ".xls" else None
        try:
            sheets = pd.read_excel(path, sheet_name=None, header=None, engine=engine)
        except Exception:
            if engine is None:
                raise
            sheets = pd.read_excel(path, sheet_name=None, header=None)  # fall back to openpyxl
        return frames_to_sheets(sheets)

    raise ValueError(f"Unsupported file type: {ext}. Upload .csv or .xlsx")


def memory_mb(sheets: dict[str, pd.DataFrame]) -> float:
    return sum(df.memory_usage(deep=True).sum() for df in sheets.values()) / (1 << 20)
