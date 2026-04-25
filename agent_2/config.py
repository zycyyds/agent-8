# -*- coding: utf-8 -*-
"""
统一配置文件

所有路径、API 配置、枚举类型、处理管线定义集中在此处。
"""
import os
from enum import Enum
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# 数据类型枚举
# ---------------------------------------------------------------------------

class DataType(Enum):
    IMAGE     = "image"
    TEXT      = "text"
    CSV       = "csv"
    EXCEL     = "excel"
    JSONL     = "jsonl"
    DIRECTORY = "directory"
    UNKNOWN   = "unknown"


# ---------------------------------------------------------------------------
# 文件扩展名
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"
}
TEXT_EXTENSIONS: Set[str] = {".txt", ".md", ".json", ".xml"}
CSV_EXTENSIONS:  Set[str] = {".csv", ".tsv"}
EXCEL_EXTENSIONS: Set[str] = {".xlsx", ".xls"}
JSONL_EXTENSIONS: Set[str] = {".jsonl"}


# ---------------------------------------------------------------------------
# 医学列名关键词（用于自动识别 CSV 列）
# ---------------------------------------------------------------------------

MEDICAL_COL_KEYWORDS = {
    "standardize": [
        "test_name", "org_name", "ab_name", "spec_type_desc",
        "diagnosis", "drug", "medication", "disease", "symptom",
        "诊断", "药品", "疾病", "症状", "检验项目", "检查项目",
        "icd", "loinc", "snomed", "atc",
    ],
    "unit_normalize": [
        "value", "result", "quantity", "dilution_value", "amount",
        "数值", "结果", "剂量", "浓度",
        "concentration", "dose", "measurement",
    ],
    "unit_col": [
        "unit", "units", "单位", "dilution_text", "dilution_comparison",
    ],
    "extract_text": [
        "text", "note", "report", "description", "comment", "narrative",
        "findings", "impression", "indication", "conclusion", "summary",
        "discharge", "radiology", "pathology", "history", "assessment",
    ],
}

# 遍历目录时跳过的文件夹名
DEFAULT_SKIP_FOLDERS: Set[str] = {"figure", "分割"}


def get_skip_folders() -> Set[str]:
    env = os.environ.get("SKIP_FOLDERS")
    if env is not None:
        if env.strip() == "":
            return set()
        return {n.strip() for n in env.split(",") if n.strip()}
    return DEFAULT_SKIP_FOLDERS.copy()


# ---------------------------------------------------------------------------
# API 配置
# ---------------------------------------------------------------------------

def get_api_config() -> Dict:
    return {
        "api_key":    "",
        "api_base":   "",
        "timeout":    int(os.environ.get("OPENAI_TIMEOUT", "120")),
        "model_name": os.environ.get("MODEL_NAME", "gpt-5.1"),
        "use_umls":   os.environ.get("USE_UMLS", "true").lower() == "true",
        "verbose":    os.environ.get("VERBOSE", "false").lower() == "true",
    }
