#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
from pathlib import Path

import pandas as pd


RAW_ROOT = Path("/Users/mkbk/PycharmProjects/step1-dataset-codegen/rawdata").resolve()
OUTPUT_ROOT = Path(r"/Users/mkbk/PycharmProjects/step1-dataset-codegen/program/output/step1_results").resolve()
RECORDS_PATH = Path(r"/Users/mkbk/PycharmProjects/step1-dataset-codegen/reorganized_output/_meta/records.json")


def safe_str(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def relative_parent_dirs(source_path: Path, raw_root: Path) -> Path:
    rel = source_path.relative_to(raw_root)
    parent = rel.parent
    return Path() if str(parent) == "." else parent


def copy_single_file(record: dict):
    source_path = Path(record["source_path"]).resolve()
    patient_id = record.get("patient_id")
    observations = record.get("observations", {})
    modality = observations.get("modality")

    if not patient_id:
        print(f"[SKIP] 非表格文件缺少 patient_id: {source_path}")
        return

    if not modality:
        print(f"[SKIP] 缺少 modality: {source_path}")
        return

    rel_parent = relative_parent_dirs(source_path, RAW_ROOT)
    dst = OUTPUT_ROOT / patient_id / modality / rel_parent / source_path.name
    ensure_parent(dst)
    shutil.copy2(source_path, dst)
    print(f"[COPY] {source_path} -> {dst}")


def split_table_file(record: dict):
    source_path = Path(record["source_path"]).resolve()
    observations = record.get("observations", {})
    modality = observations.get("modality")
    should_split = observations.get("should_split", False)
    id_column = observations.get("id_column")

    if modality != "table":
        print(f"[SKIP] 非表格记录误入 split_table_file: {source_path}")
        return

    if not should_split:
        print(f"[SKIP] 表格未标记拆分: {source_path}")
        return

    if not id_column:
        print(f"[SKIP] 表格缺少 id_column: {source_path}")
        return

    rel_parent = relative_parent_dirs(source_path, RAW_ROOT)

    try:
        workbook = pd.read_excel(source_path, sheet_name=None)
    except Exception as e:
        print(f"[ERROR] 读取表格失败: {source_path} | {e}")
        return

    for sheet_name, df in workbook.items():
        if df is None or df.empty:
            print(f"[SKIP] 空工作表: {source_path} | sheet={sheet_name}")
            continue

        matched_col = None
        for col in df.columns:
            if str(col).strip() == str(id_column).strip():
                matched_col = col
                break

        if matched_col is None:
            print(f"[SKIP] 工作表未找到 id 列: {source_path} | sheet={sheet_name} | id_column={id_column}")
            continue

        work_df = df.copy()
        work_df[matched_col] = work_df[matched_col].apply(safe_str)
        work_df = work_df[work_df[matched_col].notna()]

        if work_df.empty:
            print(f"[SKIP] 工作表无有效患者ID: {source_path} | sheet={sheet_name}")
            continue

        for patient_id, sub_df in work_df.groupby(matched_col, dropna=True):
            patient_id = safe_str(patient_id)
            if not patient_id:
                continue

            base_name = source_path.stem
            safe_sheet = str(sheet_name).strip().replace("/", "_").replace("\\", "_")
            out_name = f"{base_name}__{safe_sheet}.csv"

            dst = OUTPUT_ROOT / patient_id / modality / rel_parent / out_name
            ensure_parent(dst)
            sub_df.to_csv(dst, index=False, encoding="utf-8-sig")
            print(f"[TABLE] {source_path} | sheet={sheet_name} | patient={patient_id} -> {dst}")


def main():
    if not RECORDS_PATH.exists():
        raise FileNotFoundError(f"records.json 不存在: {RECORDS_PATH}")

    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("records.json 格式错误：顶层应为列表")

    for record in records:
        source_path = Path(record["source_path"]).resolve()
        observations = record.get("observations", {})
        modality = observations.get("modality")

        if not source_path.exists():
            print(f"[SKIP] 源文件不存在: {source_path}")
            continue

        if modality == "table":
            split_table_file(record)
        else:
            copy_single_file(record)

    print("[DONE] 重组完成")


if __name__ == "__main__":
    main()