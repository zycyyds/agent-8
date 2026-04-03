# -*- coding: utf-8 -*-
"""
数据类型检测工具

自动识别输入数据的类型（图片/文本/CSV），
返回对应的处理流程建议。
"""
import os
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import sys
# 确保可以导入配置
module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from config.settings import (
    DataType, 
    IMAGE_EXTENSIONS, 
    TEXT_EXTENSIONS, 
    CSV_EXTENSIONS,
    PROCESSING_PIPELINE,
    ProcessingStage
)


def detect_data_type(input_path: str) -> Tuple[DataType, Dict[str, Any]]:
    """
    检测输入数据的类型
    
    Args:
        input_path: 输入文件路径或文本内容
        
    Returns:
        Tuple of (DataType, metadata_dict)
        metadata_dict 包含:
        - file_path: 文件路径（如果是文件）
        - file_name: 文件名
        - extension: 文件扩展名
        - exists: 文件是否存在
        - is_directory: 是否为目录
        - processing_stages: 建议的处理阶段列表
    """
    metadata = {
        "file_path": None,
        "file_name": None,
        "extension": None,
        "exists": False,
        "is_directory": False,
        "processing_stages": [],
        "detected_type": DataType.UNKNOWN
    }
    
    # 检查是否为文件路径
    if os.path.exists(input_path):
        metadata["file_path"] = input_path
        metadata["exists"] = True
        
        if os.path.isdir(input_path):
            metadata["is_directory"] = True
            # 目录暂时归类为未知
            data_type = DataType.UNKNOWN
        else:
            metadata["file_name"] = os.path.basename(input_path)
            ext = Path(input_path).suffix.lower()
            metadata["extension"] = ext
            
            # 根据扩展名判断类型
            if ext in IMAGE_EXTENSIONS:
                data_type = DataType.IMAGE
            elif ext in CSV_EXTENSIONS:
                data_type = DataType.CSV
            elif ext in TEXT_EXTENSIONS:
                data_type = DataType.TEXT
            else:
                # 尝试读取文件内容判断
                data_type = _detect_by_content(input_path)
    else:
        # 不存在的路径，可能是直接传入的文本
        if _looks_like_path(input_path):
            # 看起来像路径但不存在
            metadata["file_path"] = input_path
            data_type = DataType.UNKNOWN
        else:
            # 纯文本内容
            data_type = DataType.TEXT
    
    metadata["detected_type"] = data_type
    metadata["processing_stages"] = [
        stage.value for stage in PROCESSING_PIPELINE.get(data_type, [])
    ]
    
    return data_type, metadata


def _looks_like_path(text: str) -> bool:
    """判断文本是否像文件路径"""
    # 简单判断：长度过长不是路径
    if len(text) > 500:
        return False
    
    # 检查是否以常见路径前缀开头
    path_prefixes = ('/', './', '../', '~/', 'C:\\', 'D:\\', 'E:\\')
    if text.startswith(path_prefixes):
        return True
    
    # 检查是否以文件扩展名结尾
    if text.endswith(tuple(IMAGE_EXTENSIONS | TEXT_EXTENSIONS | CSV_EXTENSIONS)):
        return True
    
    # 如果包含空格或逗号，更可能是文本而不是路径
    if ' ' in text or ',' in text or ':' in text:
        return False
    
    # 如果包含中文字符，可能是医学文本
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        return False
    
    # 检查是否像单位 (如 mg/dL, 10^9/L)
    import re
    if re.search(r'\d+[./×]\d+|\d+\^\d+|mg/|/L|/mm|/μL', text):
        return False
    
    # 如果包含路径分隔符且没有上述特征，可能是路径
    if os.sep in text or '/' in text:
        return True
    
    return False


def _detect_by_content(file_path: str) -> DataType:
    """
    通过文件内容判断类型
    
    Args:
        file_path: 文件路径
        
    Returns:
        DataType
    """
    try:
        # 先尝试作为文本读取
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = f.read(1024)
        
        # 检查是否像CSV（包含逗号分隔的多列）
        if ',' in first_lines:
            lines = first_lines.split('\n')
            if len(lines) > 1:
                first_line_cols = len(lines[0].split(','))
                second_line_cols = len(lines[1].split(',')) if lines[1] else 0
                if first_line_cols > 2 and first_line_cols == second_line_cols:
                    return DataType.CSV
        
        return DataType.TEXT
        
    except UnicodeDecodeError:
        # 无法作为文本读取，可能是二进制（图片）
        return DataType.IMAGE
    except Exception:
        return DataType.UNKNOWN


def get_processing_pipeline(data_type: DataType) -> list:
    """
    获取数据类型对应的处理流程
    
    Args:
        data_type: 数据类型
        
    Returns:
        处理阶段列表
    """
    return PROCESSING_PIPELINE.get(data_type, [])


def format_detection_result(data_type: DataType, metadata: Dict) -> str:
    """
    格式化检测结果为人类可读的字符串
    
    Args:
        data_type: 数据类型
        metadata: 元数据
        
    Returns:
        格式化的字符串
    """
    lines = [
        f"数据类型检测结果:",
        f"  - 类型: {data_type.value}",
    ]
    
    if metadata.get("file_path"):
        lines.append(f"  - 文件路径: {metadata['file_path']}")
    if metadata.get("file_name"):
        lines.append(f"  - 文件名: {metadata['file_name']}")
    if metadata.get("extension"):
        lines.append(f"  - 扩展名: {metadata['extension']}")
    
    lines.append(f"  - 文件存在: {'是' if metadata.get('exists') else '否'}")
    
    if metadata.get("processing_stages"):
        stages = ", ".join(metadata["processing_stages"])
        lines.append(f"  - 处理流程: {stages}")
    
    return "\n".join(lines)


# 导出供Agent使用的工具函数
def detect_input_type(input_path_or_text: str) -> Dict[str, Any]:
    """
    检测输入类型的工具函数（供Agent调用）
    
    Args:
        input_path_or_text: 输入文件路径或文本
        
    Returns:
        包含检测结果的字典:
        - type: 数据类型 (image/text/csv/unknown)
        - metadata: 元数据
        - processing_stages: 建议的处理阶段
        - description: 人类可读的描述
    """
    data_type, metadata = detect_data_type(input_path_or_text)
    
    return {
        "type": data_type.value,
        "metadata": metadata,
        "processing_stages": metadata.get("processing_stages", []),
        "description": format_detection_result(data_type, metadata)
    }
