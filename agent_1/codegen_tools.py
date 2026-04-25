import json
import re
from pathlib import Path

import pandas as pd

from agent_1.codegen_visual import analyze_visual_document


TABULAR_SUFFIXES = {".csv", ".xls", ".xlsx", ".tsv"}
PATIENT_ID_PATTERN = re.compile(r"patient[-_ ]?(\d+)", re.IGNORECASE)
NUMERIC_ID_PATTERN = re.compile(r"(?<!\d)(\d{4,})(?:\(\d+\))?(?!\d)")
DEFAULT_OBSERVATIONS = {
    "modality": "ocr",
    "should_split": False,
    "id_column": None,
}
_STEP1_RECORDS_PATH: Path | None = None


def classify_tabular_suffix(filename: str) -> str | None:
    lower_name = filename.lower()
    for suffix in TABULAR_SUFFIXES:
        if lower_name.endswith(suffix):
            return "table"
    return None


def _match_patient_id(text: str) -> str | None:
    match = PATIENT_ID_PATTERN.search(text or "")
    if not match:
        return None
    return f"patient-{match.group(1)}"


def extract_patient_id(source_path: str, content_text: str = "") -> str | None:
    path_match = _match_patient_id(source_path)
    if path_match:
        return path_match

    numeric_path_match = NUMERIC_ID_PATTERN.search(source_path or "")
    if numeric_path_match:
        return numeric_path_match.group(1)

    content_match = _match_patient_id(content_text)
    if content_match:
        return content_match

    numeric_content_match = NUMERIC_ID_PATTERN.search(content_text or "")
    if numeric_content_match:
        return numeric_content_match.group(1)
    return None


def set_step1_records_path(records_path: Path | str | None) -> None:
    global _STEP1_RECORDS_PATH
    _STEP1_RECORDS_PATH = Path(records_path) if records_path else None


def clear_step1_records_path() -> None:
    global _STEP1_RECORDS_PATH
    _STEP1_RECORDS_PATH = None


def _records_path() -> Path:
    if _STEP1_RECORDS_PATH is not None:
        return _STEP1_RECORDS_PATH
    return Path.cwd() / "reorganized_output" / "_meta" / "records.json"


def _load_records() -> list[dict]:
    path = _records_path()
    if not path.exists():
        return []
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _write_records(records: list[dict]) -> None:
    path = _records_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _default_record(file_path: str) -> dict:
    path = Path(file_path)
    patient_match = re.match(r"^(\d+)", path.stem)
    patient_id = patient_match.group(1) if patient_match else extract_patient_id(file_path, "")
    return {
        "source_path": str(path),
        "source_name": path.name,
        "patient_id": patient_id,
        "observations": dict(DEFAULT_OBSERVATIONS),
    }


def _upsert_record(file_path: str, updater) -> dict:
    source_path = str(Path(file_path))
    records = _load_records()
    target = None
    for item in records:
        if item.get("source_path") == source_path:
            target = item
            break
    if target is None:
        target = _default_record(file_path)
        records.append(target)
    target.setdefault("observations", dict(DEFAULT_OBSERVATIONS))
    target["observations"] = {
        "modality": target["observations"].get("modality", "ocr"),
        "should_split": bool(target["observations"].get("should_split", False)),
        "id_column": target["observations"].get("id_column"),
    }
    updater(target)
    _write_records(records)
    return target


def record_source_file(file_path: str) -> dict:
    record = _upsert_record(
        file_path,
        lambda item: item.update(_default_record(file_path)),
    )
    return {
        "source_path": record["source_path"],
        "source_name": record["source_name"],
        "patient_id": record["patient_id"],
    }


def _read_tabular_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path, dtype=str, keep_default_na=False)
    if file_path.suffix.lower() == ".tsv":
        return pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
    return pd.read_excel(file_path, dtype=str, header=None)


def _normalize_tabular_dataframe(file_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    if file_path.suffix.lower() in {".xls", ".xlsx"} and not df.empty:
        first_col_values = df.iloc[:, 0].fillna("").astype(str).str.strip()
        if first_col_values.ne("").all():
            normalized = df.copy()
            normalized.columns = [f"col_{idx}" for idx in range(df.shape[1])]
            normalized.insert(0, "id", first_col_values)
            return normalized
    return df


def _detect_id_column(df: pd.DataFrame) -> str | None:
    preferred_names = {"id", "patient_id", "patientid", "病例号", "病人id", "患者id", "患者编号", "病历号"}
    for column in df.columns:
        normalized = str(column).strip().lower().replace(" ", "").replace("_", "")
        if normalized in preferred_names:
            return str(column)
    for column in df.columns:
        series = df[column].fillna("").astype(str).str.strip()
        candidate_values = [value for value in series if value and value.lower() not in {"nan", "none"}]
        if not candidate_values:
            continue
        extracted = [extract_patient_id("", value) for value in candidate_values]
        non_empty = [value for value in extracted if value]
        if non_empty and len(non_empty) == len(candidate_values) and len(set(non_empty)) >= max(1, len(non_empty) // 2):
            return str(column)
    return None


def record_file_modality(file_path: str) -> dict:
    path = Path(file_path)
    if classify_tabular_suffix(path.name) == "table":
        modality = "table"
    else:
        results = analyze_visual_document(file_path)
        first = results[0] if results else {}
        modality = first.get("classification_modality", "ocr")

    record = _upsert_record(
        file_path,
        lambda item: item["observations"].update({"modality": modality}),
    )
    return {"modality": record["observations"]["modality"]}


def record_table_split_strategy(file_path: str) -> dict:
    path = Path(file_path)
    should_split = False
    id_column = None
    if classify_tabular_suffix(path.name) == "table":
        df = _normalize_tabular_dataframe(path, _read_tabular_file(path))
        id_column = _detect_id_column(df)
        should_split = id_column is not None

    record = _upsert_record(
        file_path,
        lambda item: item["observations"].update(
            {
                "should_split": should_split,
                "id_column": id_column,
            }
        ),
    )
    return {
        "should_split": record["observations"]["should_split"],
        "id_column": record["observations"]["id_column"],
    }


def _load_records_from_path(records_path: str) -> tuple[list[dict], list[str]]:
    path = Path(records_path)
    if not path.exists():
        return [], ["records.json 不存在"]
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], ["records.json 不是合法 JSON 列表"]
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        return [], ["records.json 不是合法 JSON 列表"]
    return parsed, []


def _iter_output_files(output_root: str) -> list[Path]:
    root = Path(output_root)
    if not root.exists():
        return []
    return [
        item
        for item in sorted(root.rglob("*"))
        if item.is_file() and not any(part.startswith(".") for part in item.parts) and "_meta" not in item.parts
    ]


def _is_legacy_observation_record(item: dict) -> bool:
    required_top_level = {"source_path", "source_name", "patient_id", "observations"}
    required_observations = {"modality", "should_split", "id_column"}
    if required_top_level.difference(item.keys()):
        return False
    observations = item.get("observations") or {}
    return not required_observations.difference(observations.keys())


def _is_current_runtime_record(item: dict) -> bool:
    required_top_level = {"source_path", "source_name", "patient_id", "modality", "observation"}
    if required_top_level.difference(item.keys()):
        return False
    observation = item.get("observation") or {}
    source_file = observation.get("source_file") or {}
    file_modality = observation.get("file_modality") or {}
    return (
        {"source_path", "source_name", "patient_id"}.issubset(source_file.keys())
        and "modality" in file_modality
        and "table_split_strategy" in observation
    )


def _validate_observation_records_shape(records: list[dict]) -> list[str]:
    issues: list[str] = []

    for item in records:
        if _is_legacy_observation_record(item) or _is_current_runtime_record(item):
            continue
        issues.append("records.json observation 结构不完整")
        break
    return issues


def validate_reorganized_contract(input_path: str, output_root: str, records_path: str) -> dict:
    records, record_issues = _load_records_from_path(records_path)
    output_files = _iter_output_files(output_root)
    issues: list[str] = list(record_issues)
    if not issues:
        issues.extend(_validate_observation_records_shape(records))
    sample_paths: list[str] = []

    for file_path in output_files[:10]:
        relative_path = file_path.relative_to(output_root)
        sample_paths.append(str(relative_path))
        if len(relative_path.parts) < 3:
            issues.append("输出路径层级至少应为 id/模态/文件，允许中间目录")
        if file_path.suffix.lower() in {".csv", ".tsv"}:
            lines = file_path.read_text(encoding="utf-8").splitlines()
            if not lines or ("," not in lines[0] and "\t" not in lines[0]):
                issues.append("表格输出缺少表头")

    if not output_files:
        issues.append("未检测到 _meta 之外的输出文件")

    return {
        "passed": len(issues) == 0,
        "issues": sorted(set(issues)),
        "records_count": len(records),
        "output_file_count": len(output_files),
        "sample_paths": sample_paths,
        "input_path": input_path,
        "output_root": output_root,
        "records_path": records_path,
    }
