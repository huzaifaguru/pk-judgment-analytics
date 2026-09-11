#!/usr/bin/env python3
"""
build_dashboard_data.py — exports a trimmed, JSON-safe slice of the cleaned
dataset for the static dashboard at docs/index.html.

Input : data/processed/clean_cases.csv
Output: docs/data.json
"""
import json
import math
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "clean_cases.csv"
OUT = ROOT / "docs" / "data.json"

COLUMNS = [
    "filename", "case_no", "case_type", "legal_category", "judge_folder",
    "year", "month", "outcome", "word_count", "num_sections", "bench_size",
    "petitioner", "respondent", "petitioner_plausible", "respondent_plausible",
]


def clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, bool):
        return v
    return v


def main():
    df = pd.read_csv(DATA)
    df = df[COLUMNS]
    records = []
    for row in df.to_dict(orient="records"):
        rec = {k: clean_value(v) for k, v in row.items()}
        if rec.get("petitioner_plausible") is False:
            rec["petitioner"] = None
        if rec.get("respondent_plausible") is False:
            rec["respondent"] = None
        rec.pop("petitioner_plausible", None)
        rec.pop("respondent_plausible", None)
        if rec.get("year") is not None:
            rec["year"] = int(rec["year"])
        if rec.get("month") is not None:
            rec["month"] = int(rec["month"])
        if rec.get("num_sections") is not None:
            rec["num_sections"] = int(rec["num_sections"])
        if rec.get("word_count") is not None:
            rec["word_count"] = int(rec["word_count"])
        if rec.get("bench_size") is not None:
            rec["bench_size"] = int(rec["bench_size"])
        records.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, separators=(",", ":")))
    print(f"Wrote {len(records)} records to {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
