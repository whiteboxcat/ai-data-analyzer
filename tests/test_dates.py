"""Dates with impossible years (typos like 1000-02-24) must never crash the analysis.

    python tests/test_dates.py
"""
import datetime as dt
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402

from analyzer import industry  # noqa: E402
from analyzer.pipeline import run  # noqa: E402
from analyzer.utils import implausible_dates, parse_dates  # noqa: E402


def analyse(path):
    with mock.patch.object(industry.requests, "get", side_effect=OSError("offline")):
        return run(path, with_trends=False, cleaned_out=Path(path).with_name("cleaned"))


def check(r, label):
    s = r["sheets"][0]
    rng = next(k for k in s["kpis"] if "range" in k["label"])
    assert rng["value"].startswith("2024"), (label, rng)               # typo left out of the range
    line = next(c for c in s["charts"] if c["type"] == "line")
    assert all(c.startswith("2024") for c in line["figure"]["categories"]), (label, line["figure"]["categories"])
    odd = [i for i in r["issues"] if i["kind"] == "implausible_dates"]
    assert odd and "1000-02-24" in odd[0]["detail"], (label, [i["title"] for i in r["issues"]])
    print(f"OK {label}: range {rng['value']} {rng['sub']}; flagged: {odd[0]['title']}")


def test_parse_dates_unit():
    s = pd.Series(["2024-01-05", "1000-02-24", "24/02/1000", "31/12/2024", None, "2999-01-01", "not a date"])
    parsed = parse_dates(s)
    assert str(parsed.dtype) == "datetime64[ns]"
    assert parsed.notna().tolist() == [True, False, False, True, False, False, False]
    assert implausible_dates(s).tolist() == [False, True, True, False, False, True, False]
    tz = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"]).tz_localize("Asia/Jakarta"))
    assert parse_dates(tz).notna().all()
    print("OK parse_dates unit")


def test_text_dates_csv():
    with tempfile.TemporaryDirectory() as d:
        rows = ["Order Date,Channel,GMV"]
        rows += [f"2024-{1 + i % 12:02d}-{1 + i % 27:02d},{['Shopee', 'Tokopedia'][i % 2]},{100000 + i * 1000}"
                 for i in range(60)]
        rows += ["1000-02-24,Shopee,5000"]
        p = Path(d) / "text_dates.csv"
        p.write_text("\n".join(rows))
        check(analyse(p), "CSV with text dates")


def test_excel_date_cells():
    with tempfile.TemporaryDirectory() as d:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Order Date", "Channel", "GMV"])
        for i in range(60):
            ws.append([dt.datetime(2024, 1 + i % 12, 1 + i % 27), ["Shopee", "Tokopedia"][i % 2], 100000 + i * 1000])
        ws.append([dt.datetime(1000, 2, 24), "Shopee", 50000])
        p = Path(d) / "excel_dates.xlsx"
        wb.save(p)
        check(analyse(p), "Excel date cells")


def test_one_bad_sheet_does_not_stop_the_rest():
    from analyzer import profiler
    real = profiler.profile_sheet

    def flaky(df, name="Sheet1"):
        if name == "Products":
            raise RuntimeError("boom")
        return real(df, name)
    with mock.patch("analyzer.pipeline.profile_sheet", side_effect=flaky), \
            mock.patch.object(industry.requests, "get", side_effect=OSError("offline")):
        r = run(Path(__file__).parent / "sample_messy.xlsx", with_trends=False)
    assert [s["sheet"] for s in r["sheets"]] == ["Sales", "Daily Traffic"]
    assert "boom" in r["failed_sheets"]["Products"]
    print("OK a failing sheet is skipped and reported:", r["failed_sheets"])


if __name__ == "__main__":
    test_parse_dates_unit()
    test_text_dates_csv()
    test_excel_date_cells()
    test_one_bad_sheet_does_not_stop_the_rest()
