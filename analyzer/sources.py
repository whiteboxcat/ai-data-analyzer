"""Read data from a Google Sheets or Google Drive link.

Two ways in, tried in this order:

1. Private (recommended for business data)
   A Google *service account* is a robot Google user. Share the sheet or file
   with its email (Viewer) and the app reads it through Google's API. Nobody
   else can open the file. Needs GOOGLE_SERVICE_ACCOUNT_JSON (the key file's
   contents) or GOOGLE_APPLICATION_CREDENTIALS (a path to the key file).

2. Public
   The file is shared as "Anyone with the link can view". No setup, but
   anyone who gets the link can open it.

Google Sheets read through the API come back as values per tab, so they never
go through a big Excel file at all. Drive files (.xlsx / .csv) are streamed to
disk with a size cap.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd
import requests

SHEET_MIME = "application/vnd.google-apps.spreadsheet"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly",
          "https://www.googleapis.com/auth/spreadsheets.readonly"]
TIMEOUT = 60
CHUNK = 1 << 20


class SourceError(ValueError):
    """A problem the user can fix (sharing, wrong link, too big)."""


@dataclass
class Source:
    name: str                                   # shown as the report title
    via: str                                    # "private" or "public"
    path: Path | None = None                    # downloaded file, or
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)  # raw tabs (header not detected yet)


# --------------------------------------------------------------------------- #
# Link parsing
# --------------------------------------------------------------------------- #
_PATTERNS = [
    ("sheet", r"docs\.google\.com/spreadsheets/(?:u/\d+/)?d/([a-zA-Z0-9_-]{20,})"),
    ("drive", r"drive\.google\.com/(?:u/\d+/)?file/d/([a-zA-Z0-9_-]{20,})"),
    ("drive", r"drive\.google\.com/(?:open|uc)\?(?:.*&)?id=([a-zA-Z0-9_-]{20,})"),
    ("drive", r"drive\.usercontent\.google\.com/download\?(?:.*&)?id=([a-zA-Z0-9_-]{20,})"),
]


def parse_link(link: str) -> tuple[str, str]:
    """Return ("sheet" | "drive", file_id). Raises SourceError for anything else."""
    link = (link or "").strip()
    for kind, pat in _PATTERNS:
        m = re.search(pat, link)
        if m:
            return kind, m.group(1)
    if "docs.google.com/document" in link or "docs.google.com/presentation" in link:
        raise SourceError("That's a Google Doc or Slides link. Paste a Google Sheets link or a Drive link to a .xlsx or .csv file.")
    if "drive.google.com/drive/folders" in link:
        raise SourceError("That's a folder link. Open the file inside it and copy that file's link instead.")
    raise SourceError("That doesn't look like a Google Sheets or Google Drive file link.")


# --------------------------------------------------------------------------- #
# Service account (private access)
# --------------------------------------------------------------------------- #
def _credentials_info() -> dict | None:
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise SourceError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON. Paste the whole key file content.")
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text())
    return None


def service_account_email() -> str | None:
    """The address to share private sheets with, or None if not configured."""
    try:
        info = _credentials_info()
    except SourceError:
        return None
    return (info or {}).get("client_email")


def _authorized_session():
    """requests.Session that signs every call with the service account."""
    info = _credentials_info()
    if not info:
        return None
    try:
        from google.auth.transport.requests import AuthorizedSession
        from google.oauth2 import service_account
    except ImportError as e:
        raise SourceError("Private Google access needs the google-auth package: pip install google-auth") from e
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return AuthorizedSession(creds)


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
def _stream_to_file(resp, dest: Path, max_bytes: int) -> int:
    size = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(CHUNK):
            size += len(chunk)
            if size > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise SourceError(f"This file is larger than {max_bytes // (1 << 20)} MB. "
                                  "Remove unused sheets or columns, or raise MAX_LINK_MB.")
            f.write(chunk)
    return size


def _filename_from_headers(resp, fallback: str) -> str:
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd) or re.search(r'filename="?([^";]+)"?', cd)
    return unquote(m.group(1)).strip() if m else fallback


def _looks_like_login_page(resp) -> bool:
    return "text/html" in resp.headers.get("Content-Type", "")


def _ext_for(name: str, mime: str | None) -> str:
    ext = Path(name).suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls", ".csv"):
        return ext
    if mime == "text/csv":
        return ".csv"
    if mime in (XLSX_MIME, "application/vnd.ms-excel"):
        return ".xlsx"
    raise SourceError(f"'{name}' isn't a spreadsheet. Link a Google Sheet, an .xlsx or a .csv file.")


def _private(kind: str, file_id: str, dest_dir: Path, max_bytes: int) -> Source:
    s = _authorized_session()
    meta = s.get(f"https://www.googleapis.com/drive/v3/files/{file_id}",
                 params={"fields": "name,mimeType,size", "supportsAllDrives": "true"}, timeout=TIMEOUT)
    if meta.status_code in (403, 404):
        raise PermissionError(meta.status_code)
    meta.raise_for_status()
    meta = meta.json()
    name, mime = meta.get("name", "Google file"), meta.get("mimeType")

    if mime == SHEET_MIME:
        return Source(name=name, via="private", frames=_read_sheet_values(s, file_id, max_bytes))
    if mime and mime.startswith("application/vnd.google-apps."):
        raise SourceError(f"'{name}' is a Google Doc/Slides/Form, not a spreadsheet.")

    if int(meta.get("size") or 0) > max_bytes:
        raise SourceError(f"'{name}' is larger than {max_bytes // (1 << 20)} MB.")
    ext = _ext_for(name, mime)
    dest = dest_dir / f"source{ext}"
    with s.get(f"https://www.googleapis.com/drive/v3/files/{file_id}",
               params={"alt": "media", "supportsAllDrives": "true"}, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        _stream_to_file(r, dest, max_bytes)
    return Source(name=name, via="private", path=dest)


def _read_sheet_values(s, sheet_id: str, max_bytes: int) -> dict[str, pd.DataFrame]:
    """Every tab of a Google Sheet as raw rows. Numbers stay numbers (no 'Rp 1.000' text)."""
    base = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}"
    info = s.get(base, params={"fields": "sheets.properties(title,sheetType,gridProperties)"}, timeout=TIMEOUT)
    info.raise_for_status()
    tabs = [p["properties"] for p in info.json().get("sheets", [])
            if p["properties"].get("sheetType", "GRID") == "GRID"]
    cells = sum((t.get("gridProperties", {}).get("rowCount", 0) * t.get("gridProperties", {}).get("columnCount", 0))
                for t in tabs)
    if cells * 12 > max_bytes * 4:  # rough guard before pulling huge sheets into memory
        raise SourceError("This Google Sheet is very large. Delete unused rows/columns or split it, then try again.")

    frames = {}
    for t in tabs:
        title = t["title"]
        r = s.get(f"{base}/values/{quote(_a1(title), safe='')}",
                  params={"valueRenderOption": "UNFORMATTED_VALUE",
                          "dateTimeRenderOption": "FORMATTED_STRING",
                          "majorDimension": "ROWS"}, timeout=TIMEOUT)
        r.raise_for_status()
        rows = r.json().get("values", [])
        if rows:
            width = max(len(row) for row in rows)
            frames[title] = pd.DataFrame([row + [None] * (width - len(row)) for row in rows]).replace("", None)
    return frames


def _a1(title: str) -> str:
    """Quote a tab name for A1 notation: My Tab -> 'My Tab'."""
    return "'" + title.replace("'", "''") + "'"


def _public(kind: str, file_id: str, dest_dir: Path, max_bytes: int) -> Source:
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 ai-data-analyzer"
    attempts = []
    if kind == "sheet":
        attempts.append(f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx")
    else:
        attempts.append(f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t")
        # a Drive link can point at a Google Sheet too
        attempts.append(f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx")

    for url in attempts:
        with sess.get(url, stream=True, timeout=TIMEOUT, allow_redirects=True) as r:
            if r.status_code in (401, 403, 404) or _looks_like_login_page(r):
                continue
            r.raise_for_status()
            name = _filename_from_headers(r, "Google Sheet.xlsx")
            ext = _ext_for(name, r.headers.get("Content-Type", "").split(";")[0])
            dest = dest_dir / f"source{ext}"
            _stream_to_file(r, dest, max_bytes)
            return Source(name=name, via="public", path=dest)
    raise PermissionError("public")


def fetch(link: str, dest_dir: str | Path, max_mb: int | None = None) -> Source:
    """Download / read the linked file. Tries private access first, then public."""
    kind, file_id = parse_link(link)
    dest_dir = Path(dest_dir)
    max_bytes = (max_mb or int(os.getenv("MAX_LINK_MB", "200"))) << 20
    email = service_account_email()

    if email:
        try:
            return _private(kind, file_id, dest_dir, max_bytes)
        except PermissionError:
            pass  # not shared with the robot; maybe it's public
        except SourceError:
            raise
        except Exception as e:
            raise SourceError(f"Google API error: {e}") from e
    try:
        return _public(kind, file_id, dest_dir, max_bytes)
    except PermissionError:
        pass
    except SourceError:
        raise
    except requests.RequestException as e:
        raise SourceError(f"Couldn't reach Google: {e.__class__.__name__}") from e

    if email:
        raise SourceError(f"The app can't open this file. Share it with {email} as Viewer, "
                          "or set it to \"Anyone with the link can view\".")
    raise SourceError("The app can't open this file. In Google, click Share and set General access to "
                      "\"Anyone with the link\" (Viewer), or set up private access (see README).")
