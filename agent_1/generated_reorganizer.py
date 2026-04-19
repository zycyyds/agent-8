import json
import shutil
from pathlib import Path

import pandas as pd


RAW_ROOT = Path("/Users/mkbk/PycharmProjects/agent-8/rawdata")
OUTPUT_ROOT = Path("reorganized_output")
RECORDS_PATH = OUTPUT_ROOT / "_meta" / "records.json"


def safe_patient_id(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_regular_file(record):
    source_path = Path(record["source_path"])
    patient_id = safe_patient_id(record.get("patient_id"))
    observations = record.get("observations", {}) or {}
    modality = observations.get("modality")

    if not patient_id or not modality:
        return {
            "status": "skipped",
            "reason": "missing patient_id or modality",
            "source": str(source_path),
        }

    if not source_path.exists():
        return {
            "status": "skipped",
            "reason": "source not found",
            "source": str(source_path),
        }

    target_dir = OUTPUT_ROOT / patient_id / modality
    ensure_dir(target_dir)
    target_path = target_dir / source_path.name
    shutil.copy2(source_path, target_path)

    return {
        "status": "copied",
        "source": str(source_path),
        "target": str(target_path),
    }


def split_table_file(record):
    source_path = Path(record["source_path"])
    observations = record.get("observations", {}) or {}
    modality = observations.get("modality")
    should_split = observations.get("should_split", False)
    id_column = observations.get("id_column")

    if modality != "table":
        return [{
            "status": "skipped",
            "reason": "not a table record",
            "source": str(source_path),
        }]

    if not source_path.exists():
        return [{
            "status": "skipped",
            "reason": "source not found",
            "source": str(source_path),
        }]

    if not should_split or not id_column:
        patient_id = safe_patient_id(record.get("patient_id"))
        if not patient_id:
            return [{
                "status": "skipped",
                "reason": "table not splittable and no patient_id",
                "source": str(source_path),
            }]
        target_dir = OUTPUT_ROOT / patient_id / modality
        ensure_dir(target_dir)
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        return [{
            "status": "copied",
            "source": str(source_path),
            "target": str(target_path),
        }]

    suffix = source_path.suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(source_path)
    elif suffix == ".xls":
        df = pd.read_excel(source_path)
    elif suffix == ".csv":
        df = pd.read_csv(source_path)
    else:
        return [{
            "status": "skipped",
            "reason": f"unsupported table format: {suffix}",
            "source": str(source_path),
        }]

    if id_column not in df.columns:
        return [{
            "status": "skipped",
            "reason": f"id_column '{id_column}' not found",
            "source": str(source_path),
        }]

    results = []
    working_df = df.copy()
    working_df[id_column] = working_df[id_column].apply(safe_patient_id)
    working_df = working_df[working_df[id_column].notna()]

    for patient_id, group in working_df.groupby(id_column, dropna=True):
        target_dir = OUTPUT_ROOT / str(patient_id) / modality
        ensure_dir(target_dir)

        if suffix == ".csv":
            out_name = f"{source_path.stem}_{patient_id}.csv"
            target_path = target_dir / out_name
            group.to_csv(target_path, index=False)
        else:
            out_name = f"{source_path.stem}_{patient_id}.xlsx"
            target_path = target_dir / out_name
            group.to_excel(target_path, index=False)

        results.append({
            "status": "split_written",
            "source": str(source_path),
            "target": str(target_path),
            "patient_id": str(patient_id),
            "rows": int(len(group)),
        })

    if not results:
        results.append({
            "status": "skipped",
            "reason": "no valid patient rows after split",
            "source": str(source_path),
        })

    return results


def main():
    if not RECORDS_PATH.exists():
        raise FileNotFoundError(f"records.json not found: {RECORDS_PATH}")

    with RECORDS_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    ensure_dir(OUTPUT_ROOT / "_meta")

    summary = {
        "total_records": len(records),
        "copied": 0,
        "split_written": 0,
        "skipped": 0,
        "details": [],
    }

    for record in records:
        observations = record.get("observations", {}) or {}
        modality = observations.get("modality")
        source_name = record.get("source_name", "")

        # 跳过明显无效的系统文件，但保留基于记录的处理逻辑
        if source_name == ".DS_Store":
            summary["skipped"] += 1
            summary["details"].append({
                "status": "skipped",
                "reason": "system file",
                "source": record.get("source_path"),
            })
            continue

        if modality == "table":
            results = split_table_file(record)
            for item in results:
                summary["details"].append(item)
                if item["status"] == "split_written":
                    summary["split_written"] += 1
                elif item["status"] == "copied":
                    summary["copied"] += 1
                else:
                    summary["skipped"] += 1
        else:
            result = copy_regular_file(record)
            summary["details"].append(result)
            if result["status"] == "copied":
                summary["copied"] += 1
            else:
                summary["skipped"] += 1

    summary_path = OUTPUT_ROOT / "_meta" / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "message": "reorganization finished",
        "summary_path": str(summary_path),
        "total_records": summary["total_records"],
        "copied": summary["copied"],
        "split_written": summary["split_written"],
        "skipped": summary["skipped"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()