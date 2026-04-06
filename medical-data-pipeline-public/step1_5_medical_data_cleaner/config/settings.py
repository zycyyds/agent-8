# -*- coding: utf-8 -*-
"""
系统配置文件

定义数据类型、处理流程和系统参数
"""
import os
from typing import Dict, List
from enum import Enum


class DataType(Enum):
    """数据类型枚举"""
    IMAGE = "image"           # 图片文件
    TEXT = "text"             # 文本文件
    CSV = "csv"               # CSV结构化数据
    UNKNOWN = "unknown"       # 未知类型


class ProcessingStage(Enum):
    """处理阶段枚举"""
    OCR = "ocr"                           # OCR文字识别（仅图片）
    PREPROCESSING = "preprocessing"        # 文本预处理（图片/文本）
    EXTRACTION = "extraction"              # 信息抽取（图片/文本）
    STANDARDIZATION = "standardization"    # 术语标准化（所有类型）
    UNIT_NORMALIZATION = "unit_normalization"  # 量纲统一（所有类型）


# 不同数据类型需要的处理阶段
PROCESSING_PIPELINE: Dict[DataType, List[ProcessingStage]] = {
    DataType.IMAGE: [
        ProcessingStage.OCR,
        ProcessingStage.PREPROCESSING,
        ProcessingStage.EXTRACTION,
        ProcessingStage.STANDARDIZATION,
    ],
    DataType.TEXT: [
        ProcessingStage.PREPROCESSING,
        ProcessingStage.EXTRACTION,
        ProcessingStage.STANDARDIZATION,
    ],
    DataType.CSV: [
        ProcessingStage.STANDARDIZATION,
    ],
}


# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

# 支持的文本格式
TEXT_EXTENSIONS = {'.txt', '.md', '.json', '.xml'}

# 支持的CSV格式
CSV_EXTENSIONS = {'.csv', '.tsv'}


# 医学数据相关的列名关键词（用于自动识别需要标准化的列）
MEDICAL_COLUMN_KEYWORDS = {
    # 术语标准化相关
    'standardization': [
        'test_name', 'org_name', 'ab_name', 'spec_type_desc',
        'diagnosis', 'drug', 'medication', 'disease', 'symptom',
        '诊断', '药品', '疾病', '症状', '检验项目', '检查项目',
        'icd', 'loinc', 'snomed', 'atc'
    ],
    # 量纲统一相关
    'unit_normalization': [
        'value', 'result', 'quantity', 'dilution_value', 'amount',
        '数值', '结果', '剂量', '浓度',
        'concentration', 'dose', 'measurement'
    ],
    # 单位列
    'unit_columns': [
        'unit', 'units', '单位', 'dilution_text', 'dilution_comparison'
    ]
}


# 遍历目录时默认跳过的文件夹名称
# 可以通过环境变量 SKIP_FOLDERS 覆盖，多个名称用逗号分隔
# 例如: export SKIP_FOLDERS="figure,分割,segments,raw"
DEFAULT_SKIP_FOLDERS = {'figure', '分割'}


def get_skip_folders() -> set:
    """
    获取需要跳过的文件夹名称集合
    
    优先使用环境变量 SKIP_FOLDERS，如果未设置则使用默认值。
    设置为空字符串表示不跳过任何文件夹。
    
    Returns:
        需要跳过的文件夹名称集合
    """
    env_value = os.environ.get("SKIP_FOLDERS")
    if env_value is not None:
        # 环境变量已设置
        if env_value.strip() == "":
            return set()  # 空字符串表示不跳过
        return {name.strip() for name in env_value.split(",") if name.strip()}
    return DEFAULT_SKIP_FOLDERS.copy()


# API配置
def get_api_config() -> Dict:
    """获取API配置"""
    return {
        'api_key': os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"),
        'api_base': os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        'timeout': int(os.environ.get("OPENAI_TIMEOUT", "120")),
        'model_name': os.environ.get("MODEL_NAME", "gpt-4.1-mini"),
        'use_umls': os.environ.get("USE_UMLS", "true").lower() == "true",
        'verbose': os.environ.get("VERBOSE", "false").lower() == "true",
    }
