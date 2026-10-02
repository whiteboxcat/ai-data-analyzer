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


def frames_to_sheets(raw_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Raw grids (no header yet), e.g. from the Google Sheets API, to clean sheets."""
    out = {}
    for name, raw in raw_frames.items():
        df = _finalize(raw)
        if not df.empty:
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
