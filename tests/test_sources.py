"""Offline tests for Google links and big files (Google is simulated).

    python tests/test_sources.py
"""
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyzer import industry, sources  # noqa: E402
from analyzer.pipeline import run  # noqa: E402

SHEET_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
XLSX = (ROOT / "tests/sample_messy.xlsx").read_bytes()


class FakeResp:
    def __init__(self, status=200, body=b"", headers=None, js=None):
        self.status_code, self._body, self.headers, self._js = status, body, headers or {}, js

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def json(self):
        return self._js

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_parse_links():
    cases = {
        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit#gid=0": ("sheet", SHEET_ID),
        f"https://docs.google.com/spreadsheets/u/1/d/{SHEET_ID}/edit?usp=sharing": ("sheet", SHEET_ID),
        f"https://drive.google.com/file/d/{SHEET_ID}/view?usp=drive_link": ("drive", SHEET_ID),
        f"https://drive.google.com/open?id={SHEET_ID}": ("drive", SHEET_ID),
    }
    for link, want in cases.items():
        assert sources.parse_link(link) == want, link
    for bad in ["https://example.com/x", "https://drive.google.com/drive/folders/abc",
                f"https://docs.google.com/document/d/{SHEET_ID}/edit"]:
        try:
            sources.parse_link(bad)
            raise AssertionError("should fail: " + bad)
        except sources.SourceError:
            pass
    print("OK parse_link")


def test_public_sheet():
    ok = FakeResp(200, XLSX, {"Content-Type": sources.XLSX_MIME,
                              "Content-Disposition": "attachment; filename*=UTF-8''Penjualan%202025.xlsx"})
    with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"GOOGLE_SERVICE_ACCOUNT_JSON": ""}), \
            mock.patch("requests.Session.get", return_value=ok):
        src = sources.fetch(f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit", d)
        assert src.via == "public" and src.name == "Penjualan 2025.xlsx" and src.path.exists()
        with mock.patch.object(industry.requests, "get", side_effect=OSError("offline")):
            r = run(src.path, name=src.name, cleaned_out=Path(d) / "cleaned")
        assert r["file"] == "Penjualan 2025.xlsx" and len(r["sheets"]) == 3
    print("OK public sheet ->", [s["sheet"] for s in r["sheets"]])


def test_not_shared():
    login = FakeResp(200, b"<html>Sign in</html>", {"Content-Type": "text/html; charset=utf-8"})
    with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"GOOGLE_SERVICE_ACCOUNT_JSON": ""}), \
            mock.patch("requests.Session.get", return_value=login):
        try:
            sources.fetch(f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit", d)
            raise AssertionError("should fail")
        except sources.SourceError as e:
            assert "Anyone with the link" in str(e)
    print("OK not-shared message")


def test_size_cap():
    big = FakeResp(200, b"x" * (3 << 20), {"Content-Type": "text/csv",
                                         "Content-Disposition": 'attachment; filename="big.csv"'})
    with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"GOOGLE_SERVICE_ACCOUNT_JSON": ""}), \
            mock.patch("requests.Session.get", return_value=big):
        try:
            sources.fetch(f"https://drive.google.com/file/d/{SHEET_ID}/view", d, max_mb=1)
            raise AssertionError("should fail")
        except sources.SourceError as e:
            assert "larger than 1 MB" in str(e)
            assert not list(Path(d).iterdir()), "partial download must be deleted"
    print("OK size cap")


class FakeAuthSession:
    """Answers like the Drive + Sheets APIs for a private Google Sheet with two tabs."""

    def get(self, url, params=None, **kw):
        if "drive/v3/files" in url and (params or {}).get("fields"):
            return FakeResp(js={"name": "Laporan Q3", "mimeType": sources.SHEET_MIME})
        if url.endswith(SHEET_ID):
            return FakeResp(js={"sheets": [
                {"properties": {"title": "Sales", "sheetType": "GRID", "gridProperties": {"rowCount": 200, "columnCount": 6}}},
                {"properties": {"title": "Chart 1", "sheetType": "OBJECT"}},
                {"properties": {"title": "Bob's tab", "sheetType": "GRID", "gridProperties": {"rowCount": 5, "columnCount": 2}}},
            ]})
        if "/values/" in url:
            if "Sales" in url:
                rng = np.random.default_rng(1)
                rows = [["Q3 report"], [], ["Date", "Channel", "Qty", "Revenue"]]
                for i in range(150):
                    rows.append([f"{1 + i % 28}/{7 + i % 3}/2025", ["Shopee", "Tokopedia", "Website"][i % 3],
                                 int(rng.integers(1, 5)), float(rng.integers(50, 500) * 1000)])
                rows[10] = rows[10][:2]  # ragged row, like the API returns for trailing blanks
                return FakeResp(js={"values": rows})
            return FakeResp(js={"values": [["Note"], ["hello"]]})
        raise AssertionError("unexpected url " + url)


def test_private_sheet():
    key = json.dumps({"client_email": "analyzer@my-proj.iam.gserviceaccount.com"})
    with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"GOOGLE_SERVICE_ACCOUNT_JSON": key}), \
            mock.patch.object(sources, "_authorized_session", return_value=FakeAuthSession()):
        assert sources.service_account_email() == "analyzer@my-proj.iam.gserviceaccount.com"
        src = sources.fetch(f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit", d)
        assert src.via == "private" and src.path is None and set(src.frames) == {"Sales", "Bob's tab"}
        with mock.patch.object(industry.requests, "get", side_effect=OSError("offline")):
            r = run(raw_frames=src.frames, name=src.name)
    sales = next(s for s in r["sheets"] if s["sheet"] == "Sales")
    roles = {c["name"]: c["role"] for c in sales["profile"]["columns"]}
    assert roles == {"Date": "date", "Channel": "category", "Qty": "metric", "Revenue": "metric"}, roles
    print("OK private sheet ->", r["file"], roles)


def test_private_not_shared_falls_back_with_hint():
    key = json.dumps({"client_email": "robot@p.iam.gserviceaccount.com"})

    class Denied:
        def get(self, *a, **k):
            return FakeResp(404)
    login = FakeResp(200, b"<html/>", {"Content-Type": "text/html"})
    with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"GOOGLE_SERVICE_ACCOUNT_JSON": key}), \
            mock.patch.object(sources, "_authorized_session", return_value=Denied()), \
            mock.patch("requests.Session.get", return_value=login):
        try:
            sources.fetch(f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit", d)
            raise AssertionError("should fail")
        except sources.SourceError as e:
            assert "robot@p.iam.gserviceaccount.com" in str(e)
    print("OK private fallback message names the robot email")


def test_big_csv():
    """~600k rows: loads with compact dtypes, stays within budget, exports a CSV zip."""
    n = 600_000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Order ID": np.arange(n) + 10_000_000,
        "Date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "Channel": rng.choice(["Shopee", "Tokopedia", "TikTok Shop", "Website"], n),
        "Qty": rng.integers(1, 6, n),
        "Net": rng.integers(50, 900, n) * 1000.0,
    })
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "big.csv"
        df.to_csv(p, index=False)
        mb = p.stat().st_size / 2 ** 20
        t = time.time()
        with mock.patch.object(industry.requests, "get", side_effect=OSError("offline")):
            r = run(p, cleaned_out=Path(d) / "cleaned", with_trends=False)
        assert r["sheets"][0]["rows"] == n and r["cleaned_file"] == "cleaned.zip"
        print(f"OK big CSV: {mb:.0f} MB file, {r['data_mb']} MB in memory, {time.time() - t:.1f}s, "
              f"export {r['cleaned_file']}")

    with mock.patch.dict(os.environ, {"MAX_DATA_MB": "5"}), tempfile.TemporaryDirectory() as d:
        p = Path(d) / "big.csv"
        df.head(200_000).to_csv(p, index=False)
        try:
            run(p, with_trends=False)
            raise AssertionError("budget should block")
        except ValueError as e:
            assert "too big" in str(e)
    print("OK memory budget message")


if __name__ == "__main__":
    test_parse_links()
    test_public_sheet()
    test_not_shared()
    test_size_cap()
    test_private_sheet()
    test_private_not_shared_falls_back_with_hint()
    test_big_csv()
