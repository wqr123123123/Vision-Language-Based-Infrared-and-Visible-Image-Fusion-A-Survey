"""Check the packaged per-image records against manuscript Tables 4 and 8."""
import csv
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent

def rows(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def rounded(value, digits):
    return Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)

def check_table_4():
    lookup = {}
    for r in rows("R1_C05_Low_Level_Results/table_4_means.csv"):
        dataset, method, metric = r["dataset"], r["method"], r["metric"]
        key = (method, dataset)
        if key not in lookup:
            lookup[key] = rows(f"R1_C05_Low_Level_Results/per_image/{method}/{dataset}.csv")
        records = lookup[key]
        assert len({x["Image"] for x in records}) == int(r["n"]), key
        value = mean(float(x[metric]) for x in records)
        assert abs(value - float(r["mean"])) < 1e-11, (method, dataset, metric)
        if r["rounding_match"] == "four_then_three_decimals":
            value = rounded(value, 4)
        assert rounded(value, 3) == Decimal(r["reported_table_4"]), r
    print(f"Table 4: {len(lookup)} method/dataset files, 456 means.")

def check_table_8():
    buckets = defaultdict(list)
    fusion = rows("R1_C08_Degradation_Benchmark/per_image_fusion_metrics.csv")
    controls = rows("R1_C08_Degradation_Benchmark/per_image_source_controls.csv")
    expected_ids = {x["sample_id"] for x in rows("R1_C08_Degradation_Benchmark/final_sample_manifest.csv")}
    assert len(expected_ids) == 127
    assert {x["sample_id"] for x in fusion} == expected_ids
    assert {x["sample_id"] for x in controls} == expected_ids
    for r in fusion + controls:
        buckets[(r["method"], r["degradation"])].append(r)
    count = 0
    for r in rows("R1_C08_Degradation_Benchmark/table_8_means.csv"):
        records = buckets[(r["method"], r["degradation"])]
        assert len({x["sample_id"] for x in records}) == int(r["n"]), r
        value = mean(float(x[r["metric"]]) for x in records)
        printed = Decimal(r["reported_table_8"])
        assert rounded(value, -printed.as_tuple().exponent) == printed, r
        count += 1
    print(f"Table 8: {len(fusion)} fusion records, {len(controls)} source-control records, {count} means.")

if __name__ == "__main__":
    check_table_4()
    check_table_8()
