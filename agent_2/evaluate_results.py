#!/usr/bin/env python3
"""Evaluate output quality of medical_data_cleaner_v2 runs.

Usage:
    python evaluate_results.py                        # latest results dir
    python evaluate_results.py results/results_xxx    # specific dir
    python evaluate_results.py --save                 # also save JSON report
    python evaluate_results.py --detail               # show drill-down diagnostics
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def find_latest_results(results_root: Path) -> Path:
    subdirs = sorted(
        [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith("results_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not subdirs:
        raise FileNotFoundError(f"No results_* directories found in {results_root}")
    return subdirs[0]


def load_files(results_dir: Path) -> dict:
    found = {}
    for f in results_dir.iterdir():
        stem = f.stem
        if f.suffix == ".json" and stem.startswith("summary"):
            found["summary"] = f
        elif f.suffix == ".csv" and stem.startswith("entities"):
            found["entities"] = f
        elif f.suffix == ".csv" and stem.startswith("patients"):
            found["patients"] = f
        elif f.suffix == ".csv" and stem.startswith("merged"):
            found["merged"] = f
    missing = [k for k in ("summary", "entities", "patients") if k not in found]
    if missing:
        raise FileNotFoundError(f"Missing output files: {missing}")
    return found


# ── evaluation modules ───────────────────────────────────────────────────────

def eval_processing(summary: dict) -> dict:
    stats = summary.get("statistics", {})
    total = stats.get("total_files", 0)
    processed = stats.get("processed_files", 0)
    failed = stats.get("failed_files", 0)

    # collect failed patient ids from summary
    failed_patients = [
        pid for pid, pdata in summary.get("patients", {}).items()
        if pdata.get("failed_count", 0) > 0
    ]

    return {
        "total_patients": stats.get("total_patients", 0),
        "total_files": total,
        "processed_files": processed,
        "failed_files": failed,
        "success_rate_pct": round(processed / total * 100, 1) if total else 0,
        "failed_patient_ids": failed_patients,
    }


def eval_entities(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {"total_entities": 0}

    per_patient = df.groupby("patient_id").size()
    category_dist = df["category"].value_counts().to_dict()

    # empty value rate
    empty_value = (df["value"].isna() | (df["value"].astype(str).str.strip() == "")).sum()

    # unit coverage for numeric-ish categories
    numeric_df = df[df["category"].isin(["LabValue", "Test"])]
    unit_cov = (
        round(numeric_df["unit"].notna().sum() / len(numeric_df) * 100, 1)
        if len(numeric_df) > 0 else None
    )

    # numeric value parse rate for LabValue/Test
    def is_numeric(v):
        try:
            float(str(v).replace(",", "").strip())
            return True
        except (ValueError, TypeError):
            return False

    numeric_parse_rate = None
    if len(numeric_df) > 0:
        parsed = numeric_df["value"].dropna().apply(is_numeric).sum()
        numeric_parse_rate = round(parsed / len(numeric_df) * 100, 1)

    # duplicate entity names per patient
    dup_count = (
        df.groupby(["patient_id", "name"]).size().gt(1).sum()
    )

    return {
        "total_entities": total,
        "category_distribution": category_dist,
        "per_patient": {
            "mean": round(per_patient.mean(), 1),
            "min": int(per_patient.min()),
            "max": int(per_patient.max()),
            "std": round(per_patient.std(), 1) if len(per_patient) > 1 else 0.0,
        },
        "empty_value_rate_pct": round(empty_value / total * 100, 1),
        "unit_coverage_pct": unit_cov,
        "numeric_value_parse_rate_pct": numeric_parse_rate,
        "duplicate_name_per_patient_count": int(dup_count),
    }


def eval_merged(df: pd.DataFrame) -> dict:
    extracted_cols = [c for c in df.columns if c.startswith("Extracted_")]
    if not extracted_cols:
        return {"extracted_columns": 0}

    filled_cols = sum(1 for c in extracted_cols if df[c].notna().any())
    fill_rate = round(filled_cols / len(extracted_cols) * 100, 1)

    missing_per_patient = df[extracted_cols].isna().mean(axis=1) * 100

    # columns with >80% missing
    sparse_cols = [
        c for c in extracted_cols
        if df[c].isna().mean() > 0.8
    ]

    return {
        "total_columns": len(df.columns),
        "extracted_columns": len(extracted_cols),
        "filled_extracted_columns": filled_cols,
        "extracted_fill_rate_pct": fill_rate,
        "per_patient_missing_pct": {
            "mean": round(missing_per_patient.mean(), 1),
            "min": round(missing_per_patient.min(), 1),
            "max": round(missing_per_patient.max(), 1),
        },
        "sparse_columns_gt80pct_missing": len(sparse_cols),
        "_sparse_col_names": sparse_cols,
    }


# ── detail diagnostics ────────────────────────────────────────────────────────

def print_detail(entities_df: pd.DataFrame, merged_df: pd.DataFrame | None, summary: dict | None = None):
    print(f"\n{SEP}")
    print("  DETAIL DIAGNOSTICS")
    print(SEP)

    # 1. Duplicate entity names per patient
    print("\n▶ Duplicate entity names (same name extracted multiple times per patient)")
    dups = (
        entities_df.groupby(["patient_id", "name"])
        .size()
        .reset_index(name="count")
        .query("count > 1")
        .sort_values(["patient_id", "count"], ascending=[True, False])
    )
    if dups.empty:
        print("  None found.")
    else:
        print(f"  {'patient_id':<12} {'name':<30} {'count':>5}")
        print(f"  {'-'*12} {'-'*30} {'-'*5}")
        for _, row in dups.iterrows():
            print(f"  {str(row['patient_id']):<12} {str(row['name']):<30} {row['count']:>5}")

    # 2. Non-numeric values in LabValue/Test
    print("\n▶ Non-numeric values in LabValue/Test entities")
    def is_numeric(v):
        try:
            float(str(v).replace(",", "").strip())
            return True
        except (ValueError, TypeError):
            return False

    numeric_df = entities_df[entities_df["category"].isin(["LabValue", "Test"])].copy()
    non_numeric = numeric_df[
        numeric_df["value"].notna() &
        ~numeric_df["value"].apply(is_numeric)
    ][["patient_id", "name", "value", "unit"]].drop_duplicates()
    if non_numeric.empty:
        print("  All values are numeric.")
    else:
        print(f"  {'patient_id':<12} {'name':<25} {'value':<20} {'unit'}")
        print(f"  {'-'*12} {'-'*25} {'-'*20} {'-'*10}")
        for _, row in non_numeric.head(30).iterrows():
            print(f"  {str(row['patient_id']):<12} {str(row['name']):<25} {str(row['value']):<20} {str(row.get('unit',''))}")
        if len(non_numeric) > 30:
            print(f"  ... and {len(non_numeric)-30} more")

    # 3. Sparse extracted columns in merged
    if merged_df is not None:
        extracted_cols = [c for c in merged_df.columns if c.startswith("Extracted_")]
        sparse = [c for c in extracted_cols if merged_df[c].isna().mean() > 0.8]
        print(f"\n▶ Sparse Extracted_* columns (>80% missing)  [{len(sparse)} total]")
        if not sparse:
            print("  None.")
        else:
            print(f"  {'column':<50} {'fill%':>6}")
            print(f"  {'-'*50} {'-'*6}")
            for c in sorted(sparse, key=lambda x: merged_df[x].notna().mean()):
                fill = round(merged_df[c].notna().mean() * 100, 1)
                print(f"  {c:<50} {fill:>5.1f}%")

        # 4. Patients with 100% missing in extracted cols
        print("\n▶ Patients with 100% missing in Extracted_* columns")
        if extracted_cols:
            id_col = merged_df.columns[0]
            fully_missing = merged_df[merged_df[extracted_cols].isna().all(axis=1)][id_col].tolist()
            if fully_missing:
                print(f"  {fully_missing}")
            else:
                print("  None.")

    # 5. Empty patient dirs in input
    print("\n▶ Empty patient directories in input")
    input_path = (summary or {}).get("plan", {}).get("input_path")
    if input_path and Path(input_path).is_dir():
        empty_dirs = [
            d.name for d in sorted(Path(input_path).iterdir())
            if d.is_dir() and d.name.isdigit() and not any(f for f in d.rglob("*") if f.is_file())
        ]
        print(f"  {empty_dirs}  (no files, skipped by pipeline)" if empty_dirs else "  None.")
    else:
        print(f"  Input path not accessible: {input_path}")

    print(f"\n{SEP}\n")


# ── report printer ────────────────────────────────────────────────────────────

SEP = "=" * 62

def _bar(count: int, max_count: int, width: int = 25) -> str:
    filled = round(count / max_count * width) if max_count else 0
    return "█" * filled + "░" * (width - filled)


def print_report(results_dir: Path, proc: dict, ent: dict, merged: dict):
    print(f"\n{SEP}")
    print(f"  EVALUATION REPORT  —  {results_dir.name}")
    print(SEP)

    # ── Processing ──
    print("\n▶ Processing")
    print(f"  Patients       : {proc['total_patients']}")
    print(f"  Files total    : {proc['total_files']}")
    print(f"  Processed      : {proc['processed_files']}")
    print(f"  Failed         : {proc['failed_files']}")
    sr = proc["success_rate_pct"]
    flag = "✓" if sr == 100 else ("⚠" if sr >= 80 else "✗")
    print(f"  Success rate   : {sr}%  {flag}")
    if proc["failed_patient_ids"]:
        print(f"  Failed patients: {proc['failed_patient_ids']}")

    # ── Entities ──
    print("\n▶ Entity Extraction")
    if ent.get("total_entities", 0) == 0:
        print("  No entities found.")
    else:
        print(f"  Total entities : {ent['total_entities']}")
        ps = ent["per_patient"]
        print(f"  Per patient    : mean={ps['mean']}  min={ps['min']}  max={ps['max']}  std={ps['std']}")
        print(f"  Empty value %  : {ent['empty_value_rate_pct']}%")
        if ent["unit_coverage_pct"] is not None:
            print(f"  Unit coverage  : {ent['unit_coverage_pct']}%  (LabValue/Test)")
        if ent["numeric_value_parse_rate_pct"] is not None:
            print(f"  Numeric parse  : {ent['numeric_value_parse_rate_pct']}%  (LabValue/Test)")
        print(f"  Dup names/pt   : {ent['duplicate_name_per_patient_count']}")

        print("\n  Category distribution:")
        cat_dist = ent["category_distribution"]
        max_cnt = max(cat_dist.values()) if cat_dist else 1
        for cat, cnt in sorted(cat_dist.items(), key=lambda x: -x[1]):
            pct = round(cnt / ent["total_entities"] * 100, 1)
            print(f"    {cat:<12} {cnt:>4}  {pct:>5.1f}%  {_bar(cnt, max_cnt)}")

    # ── Merged ──
    print("\n▶ Merged Data")
    if not merged or merged.get("extracted_columns", 0) == 0:
        print("  No merged file or no Extracted_* columns.")
    else:
        print(f"  Total columns  : {merged['total_columns']}")
        print(f"  Extracted cols : {merged['extracted_columns']}")
        print(f"  Fill rate      : {merged['extracted_fill_rate_pct']}%")
        mp = merged["per_patient_missing_pct"]
        print(f"  Missing/patient: mean={mp['mean']}%  min={mp['min']}%  max={mp['max']}%")
        print(f"  Sparse cols    : {merged['sparse_columns_gt80pct_missing']}  (>80% missing)")

    print(f"\n{SEP}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate medical_data_cleaner_v2 results")
    parser.add_argument("results_dir", nargs="?", help="Path to results_* directory (default: latest)")
    parser.add_argument("--save", action="store_true", help="Save evaluation_report.json into results dir")
    parser.add_argument("--detail", action="store_true", help="Show drill-down diagnostics")
    args = parser.parse_args()

    base = Path(__file__).parent / "results"

    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = Path.cwd() / results_dir
    else:
        results_dir = find_latest_results(base)

    print(f"Loading results from: {results_dir}")
    files = load_files(results_dir)

    with open(files["summary"], encoding="utf-8") as f:
        summary = json.load(f)

    entities_df = pd.read_csv(files["entities"])
    merged_df = pd.read_csv(files["merged"]) if "merged" in files else None

    proc = eval_processing(summary)
    ent = eval_entities(entities_df)
    merged = eval_merged(merged_df) if merged_df is not None else {}

    print_report(results_dir, proc, ent, merged)

    if args.detail:
        print_detail(entities_df, merged_df, summary)

    if args.save:
        report = {"processing": proc, "entities": ent, "merged": merged}
        out = results_dir / "evaluation_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
