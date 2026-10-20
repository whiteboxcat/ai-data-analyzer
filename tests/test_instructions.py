"""The optional "cleaning instructions" box (fake AI, no keys needed):
    python tests/test_instructions.py
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from analyzer import ai_client, industry, instructions  # noqa: E402
from analyzer.pipeline import run  # noqa: E402
from tests.test_pipeline import fake_call as base_fake  # noqa: E402
from tests.test_workbook import build  # noqa: E402

TEXT = """Remove rows where extract_version is RERUN.
Keep the latest inventory snapshot per distributor and SKU.
Don't standardise SKU names in offline_sell_in, we report raw codes there.
Drop the region column in sales_target.
Convert currency to USD."""

OPS = {
    "operations": [
        {"op": "drop_rows", "sheet": "offline_sell_in", "description": "Remove RERUN rows",
         "filters": [{"column": "extract_version", "op": "==", "value": "rerun"}]},
        {"op": "dedupe", "sheet": "inventory", "keys": ["distributor", "sku"], "keep": "last",
         "order_by": "snapshot_ts", "description": "Keep the latest inventory snapshot"},
        {"op": "skip_auto_fix", "kinds": ["mapped_names"], "sheet": "offline_sell_in", "column": "sku_raw",
         "description": "Don't standardise SKU names in offline_sell_in"},
        {"op": "drop_columns", "sheet": "sales_target", "columns": ["region"],
         "description": "Drop region in sales_target"},                      # sheet doesn't exist
        {"op": "drop_rows", "sheet": "inventory", "description": "Remove everything",
         "filters": [{"column": "units", "op": ">=", "value": 0}]},          # would empty the sheet
    ],
    "not_possible": [{"instruction": "Convert currency to USD", "reason": "No exchange rate in the data."}],
    "summary": "Removed reruns, kept the latest snapshots, left SKU names as written.",
}


def fake_call(system, user, max_tokens, model=None):
    if system.startswith("You turn a user's data-cleaning"):
        assert "Remove rows where extract_version is RERUN" in user
        return json.dumps(OPS)
    return base_fake(system, user, max_tokens, model)


def add_inventory(path: Path):
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb.create_sheet("inventory")
    ws.append(["snapshot_ts", "distributor", "sku", "units", "status"])
    for ts, d, s, u in [("2026-06-30 18:00", "A", "FBL200", 50), ("2026-06-30 09:00", "A", "FBL200", 40),
                        ("2026-06-29 18:00", "A", "HTTW450", 30), ("2026-07-01 08:00", "B", "FBL200", 20),
                        ("2026-06-01 08:00", "B", "FBL200", 10)]:
        ws.append([ts, d, s, u, "FINAL"])
    wb.save(path)


def test_pipeline(tmp: Path):
    p = tmp / "bank.xlsx"
    build(p)
    add_inventory(p)
    with mock.patch.dict(ai_client._CALLERS, {"claude": fake_call}), \
         mock.patch.object(industry.requests, "get", side_effect=OSError("offline")):
        r = run(p, provider="claude", context={"cleaning_instructions": TEXT}, with_trends=False)
    uc = r["user_cleaning"]
    st = [s["status"] for s in uc["steps"]]
    assert st == ["done", "done", "done", "failed", "failed", "not_possible"], uc["steps"]
    assert "Removed 1 row" in uc["steps"][0]["detail"], uc["steps"][0]
    assert "kept the latest by snapshot_ts" in uc["steps"][1]["detail"]
    assert "every row" in uc["steps"][4]["detail"]
    assert uc["skip"] == {"offline_sell_in": ["mapped_names@sku_raw"]}

    inv = next(s for s in r["sheets"] if s["sheet"] == "inventory")
    assert inv["rows"] == 3, inv["rows"]                       # A/FBL200 18:00, A/HTTW450, B/FBL200 07-01
    sell = next(s for s in r["sheets"] if s["sheet"] == "offline_sell_in")
    cols = [c["name"] for c in sell["profile"]["columns"]]
    assert "sku_canonical" not in cols, cols                     # user switched the lookup off here
    assert "distributor_canonical" in cols, cols                 # ...but only for the SKU column
    m = {i["column"]: i for i in sell["issues"] if i["kind"] == "mapped_names"}
    assert m["sku_raw"]["user_skipped"] and not m["sku_raw"]["auto_fix"]
    assert "Left as written" in m["sku_raw"]["recommendation"] and "new 'sku_canonical'" not in m["sku_raw"]["detail"]
    assert m["distributor_raw"]["auto_fix"] and not m["distributor_raw"].get("user_skipped")
    assert "reloaded_rows" not in {i["kind"] for i in sell["issues"]}    # already removed by the user
    assert sell["rows"] == 12
    json.dumps(r)
    print("OK pipeline: user steps run first, failures reported, a switched-off fix stays off")


def test_no_ai_and_blank():
    sheets = {"s": pd.DataFrame({"a": [1, 2]})}
    assert instructions.run(sheets, "   ", "claude") is None
    rep = instructions.run(sheets, "drop row 1", None)
    assert rep["error"] and "AI model" in rep["error"] and len(sheets["s"]) == 2
    print("OK blank box changes nothing; without an AI model nothing is applied and the report says why")


def test_ops_directly():
    df = pd.DataFrame({"status": ["Paid", "CANCELLED", "paid", None], "channel": ["Shopee", "shopee ", "Tokped", "x"],
                       "qty": ["1", "2", "3", ""]})
    df.attrs["side_tables"] = [{"columns": "J–K"}]
    sheets = {"orders": df}
    from analyzer.profiler import profile_sheet
    profiles = {"orders": profile_sheet(df, "orders")}
    skip = {}
    msg = instructions._apply({"op": "drop_rows", "sheet": "Orders",
                               "filters": [{"column": "status", "op": "==", "value": "cancelled"}]}, sheets, profiles, skip)
    assert "Removed 1 row" in msg and len(sheets["orders"]) == 3
    assert sheets["orders"].attrs["side_tables"]                 # notes about the sheet survive
    msg = instructions._apply({"op": "rename_values", "sheet": "orders", "column": "channel",
                               "mapping": {"tokped": "Tokopedia"}}, sheets, profiles, skip)
    assert sheets["orders"]["channel"].tolist() == ["Shopee", "Tokopedia", "x"], sheets["orders"]
    instructions._apply({"op": "fill_missing", "sheet": "orders", "column": "status", "value": "Pending"},
                        sheets, profiles, skip)
    assert sheets["orders"]["status"].tolist() == ["Paid", "paid", "Pending"]
    instructions._apply({"op": "set_type", "sheet": "orders", "column": "qty", "type": "number"}, sheets, profiles, skip)
    assert sheets["orders"]["qty"].tolist()[:2] == [1, 3]
    instructions._apply({"op": "skip_auto_fix", "kinds": ["missing_values", "nonsense"]}, sheets, profiles, skip)
    assert skip == {"*": {"missing_values"}}
    instructions._apply({"op": "skip_auto_fix", "kinds": ["whitespace"], "sheet": "orders", "column": "CHANNEL"},
                        sheets, profiles, skip)
    rep = {"skip": {k: sorted(v) for k, v in skip.items()}}
    assert instructions.is_skipped(rep, "orders", "whitespace", "channel")
    assert not instructions.is_skipped(rep, "orders", "whitespace", "status")
    assert instructions.is_skipped(rep, "other", "missing_values", "x")
    try:
        instructions._apply({"op": "keep_rows", "sheet": "orders", "filters": [{"column": "nope", "value": 1}]},
                            sheets, profiles, skip)
        raise AssertionError("unknown column must fail")
    except instructions.QueryError as e:
        assert "Unknown" in str(e)
    print("OK operations: drop/keep rows, rename, fill, set type, switch off a fix; bad columns refused")


if __name__ == "__main__":
    test_no_ai_and_blank()
    test_ops_directly()
    with tempfile.TemporaryDirectory() as d:
        test_pipeline(Path(d))
