"""End-to-end pipeline:

    load (all sheets) -> profile -> detect issues -> clean -> re-profile
    -> AI plan (charts + industry) -> Python builds charts -> AI insights
    -> industry trends from live news
"""
from __future__ import annotations

import gc
import logging
import os
import time
import zipfile
from pathlib import Path

import pandas as pd

from . import charts as ch
from . import cleaner, industry, insights, planner
from .loader import frames_to_sheets, load_file, memory_mb
from .profiler import profile_sheet
from .utils import to_jsonable


log = logging.getLogger(__name__)

EXCEL_MAX_ROWS = 1_048_000          # Excel's sheet limit (minus a little headroom)
EXCEL_MAX_CELLS = 1_000_000         # beyond this, writing .xlsx is slow (~10s per million cells)


def data_budget_mb() -> float:
    """How much loaded data we accept. Processing needs roughly 3x this in RAM.
    Default: no limit locally, 120 MB on Render's free 512 MB instance."""
    v = os.getenv("MAX_DATA_MB")
    if v:
        return float(v)
    return 120.0 if os.getenv("RENDER") else 0.0


def run(path: str | Path | None = None, provider: str | None = None, context: dict | None = None,
        auto_clean: bool = True, with_trends: bool = True, cleaned_out: str | Path | None = None,
        raw_frames: dict[str, pd.DataFrame] | None = None, name: str | None = None) -> dict:
    """Analyse a file on disk (`path`) or raw grids already in memory (`raw_frames`,
    e.g. tabs read from the Google Sheets API)."""
    context = context or {}
    t0 = time.time()
    timings = {}

    sheets = frames_to_sheets(raw_frames) if raw_frames is not None else load_file(path)
    raw_frames = None
    if not sheets:
        raise ValueError("No data found in the file.")
    size = memory_mb(sheets)
    budget = data_budget_mb()
    if budget and size > budget:
        rows = sum(len(d) for d in sheets.values())
        raise ValueError(f"This data is too big for the server's memory ({rows:,} rows, about {size:,.0f} MB "
                         f"loaded; the limit is {budget:,.0f} MB). Remove unused columns or sheets, "
                         "split it by period, or move to a larger Render plan and raise MAX_DATA_MB.")
    sheet_names = list(sheets)

    # 1. profile + quality checks on the RAW data, then clean.
    # A sheet that fails is reported and skipped; it never stops the other sheets.
    all_issues, clean_logs, cleaned, profiles, failed = [], {}, {}, [], {}
    for sname in sheet_names:
        df = sheets.pop(sname)               # drop the raw copy as soon as it's cleaned
        try:
            raw_profile = profile_sheet(df, sname)
            iss = cleaner.detect_issues(df, raw_profile)
            if auto_clean:
                clean_df, log_lines = cleaner.apply_fixes(df, raw_profile, iss)
            else:
                clean_df, log_lines = df, []
            # 2. re-profile the cleaned data (types are now correct)
            profiles.append(profile_sheet(clean_df, sname))
            cleaned[sname], clean_logs[sname] = clean_df, log_lines
            all_issues += iss
        except Exception as e:  # noqa: BLE001 - report any sheet-level failure
            log.exception("sheet %s failed", sname)
            failed[sname] = f"{e.__class__.__name__}: {e}"
        del df
        gc.collect()
    if not cleaned:
        first = next(iter(failed.values()), "no data")
        raise ValueError(f"No sheet could be analysed. {first}")
    timings["profile_clean"] = round(time.time() - t0, 2)

    # 3. AI plan
    plan = planner.plan(profiles, all_issues, provider, context)
    timings["plan"] = round(time.time() - t0, 2)

    # 4. Python executes the charts
    sheet_results = []
    for p in profiles:
        if not p["rows"]:
            continue
        df = cleaned[p["sheet"]]
        sp = plan["sheets"].get(p["sheet"], {"charts": []})
        built = []
        for i, spec in enumerate(sp["charts"]):
            try:
                c = ch.build_chart(df, spec, p)
                c["id"] = f"{p['sheet']}::{i}"
                if c["figure"]["series"]:
                    built.append(c)
            except Exception as e:  # never let one bad chart kill the page
                plan["rejected"].append({"sheet": p["sheet"], "spec": spec, "reason": f"build error: {e}"})
        sheet_results.append({
            "sheet": p["sheet"], "rows": p["rows"], "columns_count": p["columns_count"],
            "description": sp.get("description"), "used_fallback": sp.get("used_fallback"),
            "kpis": ch.kpis(df, p), "charts": built, "profile": p,
            "issues": [i for i in all_issues if i["sheet"] == p["sheet"]],
            "clean_log": clean_logs.get(p["sheet"], []),
            "preview": to_jsonable(df.head(15).astype(object).where(df.head(15).notna(), None)
                                   .to_dict(orient="split")),
        })
    timings["charts"] = round(time.time() - t0, 2)

    # 5. insights
    ins = insights.generate(plan, sheet_results, all_issues, context, provider)
    for s in sheet_results:
        for c in s["charts"]:
            c["insight"] = (ins.get("chart_insights") or {}).get(c["id"])
    timings["insights"] = round(time.time() - t0, 2)

    # 6. industry trends
    highlights = {"summary": ins.get("executive_summary") or plan.get("workbook_summary"),
                  "kpis": {s["sheet"]: s["kpis"] for s in sheet_results}}
    trends = industry.run(plan.get("industry"), provider, highlights, context.get("language"), with_trends)
    timings["trends"] = round(time.time() - t0, 2)

    # 7. cleaned data for download (.xlsx, or a .zip of CSVs when too big for Excel)
    cleaned_file = save_cleaned(cleaned, all_issues, clean_logs, cleaned_out) if cleaned_out else None

    # skipped sheets (blank)
    empty = [n for n in sheet_names if n not in {s["sheet"] for s in sheet_results} and n not in failed]
    return to_jsonable({
        "file": name or (Path(path).name if path else "Data"), "provider": provider, "auto_clean": auto_clean,
        "cleaned_file": cleaned_file, "data_mb": round(size, 1),
        "workbook_summary": plan.get("workbook_summary"), "industry": plan.get("industry"),
        "plan_source": plan["source"], "plan_error": plan.get("error"), "rejected_charts": plan["rejected"],
        "sheets": sheet_results, "empty_sheets": empty, "failed_sheets": failed, "issues": all_issues,
        "insights": ins, "trends": trends, "timings": timings,
    })


def save_cleaned(cleaned: dict[str, pd.DataFrame], issues, logs, out_path) -> str:
    """Write the cleaned data. Returns the file name actually written."""
    out_path = Path(out_path)
    rows = [{"Sheet": i["sheet"], "Column": i["column"], "Severity": i["severity"], "Issue": i["title"],
             "Recommendation": i["recommendation"], "Auto-fixed": "yes" if i["auto_fix"] else "no"}
            for i in issues]
    rows += [{"Sheet": s, "Issue": "Applied fix", "Recommendation": l} for s, ls in logs.items() for l in ls]
    report = pd.DataFrame(rows)

    too_big = any(len(df) > EXCEL_MAX_ROWS for df in cleaned.values()) or \
        sum(df.size for df in cleaned.values()) > EXCEL_MAX_CELLS
    if too_big:
        zpath = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for sname, df in cleaned.items():
                with z.open(f"{_safe(sname)}.csv", "w") as f:
                    df.to_csv(f, index=False)
            with z.open("_cleaning_report.csv", "w") as f:
                report.to_csv(f, index=False)
        return zpath.name

    xpath = out_path.with_suffix(".xlsx")
    with pd.ExcelWriter(xpath) as xw:
        for sname, df in cleaned.items():
            df.to_excel(xw, sheet_name=_safe(sname)[:31], index=False)
        report.to_excel(xw, sheet_name="_Cleaning report", index=False)
    return xpath.name


def _safe(name: str) -> str:
    return "".join(ch if ch not in '[]:*?/\\' else "_" for ch in str(name)).strip() or "Sheet"
