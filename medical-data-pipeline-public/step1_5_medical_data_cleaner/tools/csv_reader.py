# -*- coding: utf-8 -*-
"""
CSV文件读取和分析工具

提供CSV文件的读取、列名分析、数据采样等功能，
供Agent自动判断哪些列需要进行标准化和量纲统一。
"""
import os
import csv
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import sys
module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from config.settings import MEDICAL_COLUMN_KEYWORDS


def read_csv_columns(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    读取CSV文件的列名信息
    
    Args:
        file_path: CSV文件路径
        encoding: 文件编码，默认utf-8
        
    Returns:
        包含列信息的字典:
        - success: 是否成功
        - columns: 列名列表
        - column_count: 列数
        - error: 错误信息（如果有）
    """
    result = {
        "success": False,
        "columns": [],
        "column_count": 0,
        "error": None
    }
    
    if not os.path.exists(file_path):
        result["error"] = f"文件不存在: {file_path}"
        return result
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.reader(f)
            columns = next(reader, None)
            
            if columns:
                result["success"] = True
                result["columns"] = columns
                result["column_count"] = len(columns)
            else:
                result["error"] = "CSV文件为空或无列名"
                
    except UnicodeDecodeError:
        # 尝试其他编码
        for enc in ['gbk', 'gb2312', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    reader = csv.reader(f)
                    columns = next(reader, None)
                    if columns:
                        result["success"] = True
                        result["columns"] = columns
                        result["column_count"] = len(columns)
                        break
            except:
                continue
        
        if not result["success"]:
            result["error"] = f"无法解析文件编码"
            
    except Exception as e:
        result["error"] = f"读取文件失败: {str(e)}"
    
    return result


def read_csv_sample(
    file_path: str, 
    sample_rows: int = 5,
    encoding: str = 'utf-8'
) -> Dict[str, Any]:
    """
    读取CSV文件的样本数据（列名 + 前几行）
    
    Args:
        file_path: CSV文件路径
        sample_rows: 采样行数，默认5行
        encoding: 文件编码
        
    Returns:
        包含样本数据的字典:
        - success: 是否成功
        - columns: 列名列表
        - sample_data: 样本数据（列表的列表）
        - row_count: 实际采样行数
        - total_rows: 总行数（预估）
        - error: 错误信息
    """
    result = {
        "success": False,
        "columns": [],
        "sample_data": [],
        "row_count": 0,
        "total_rows": 0,
        "error": None
    }
    
    if not os.path.exists(file_path):
        result["error"] = f"文件不存在: {file_path}"
        return result
    
    try:
        # 获取文件行数估计
        with open(file_path, 'r', encoding=encoding) as f:
            # 快速统计行数
            result["total_rows"] = sum(1 for _ in f) - 1  # 减去表头
        
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.reader(f)
            columns = next(reader, None)
            
            if not columns:
                result["error"] = "CSV文件为空或无列名"
                return result
            
            result["columns"] = columns
            
            # 读取样本数据
            for i, row in enumerate(reader):
                if i >= sample_rows:
                    break
                result["sample_data"].append(row)
            
            result["row_count"] = len(result["sample_data"])
            result["success"] = True
            
    except UnicodeDecodeError:
        for enc in ['gbk', 'gb2312', 'latin-1']:
            try:
                return read_csv_sample(file_path, sample_rows, enc)
            except:
                continue
        result["error"] = f"无法解析文件编码"
    except Exception as e:
        result["error"] = f"读取文件失败: {str(e)}"
    
    return result


def analyze_columns_for_standardization(
    columns: List[str],
    sample_data: Optional[List[List[str]]] = None
) -> Dict[str, Any]:
    """
    分析列名，判断哪些列可能需要标准化或量纲统一
    
    Args:
        columns: 列名列表
        sample_data: 可选的样本数据，用于辅助判断
        
    Returns:
        分析结果字典:
        - standardization_candidates: 可能需要术语标准化的列
        - unit_normalization_candidates: 可能需要量纲统一的列
        - unit_columns: 单位相关的列
        - analysis_details: 详细分析信息
    """
    result = {
        "standardization_candidates": [],
        "unit_normalization_candidates": [],
        "unit_columns": [],
        "analysis_details": []
    }
    
    columns_lower = [col.lower() for col in columns]
    
    for i, col in enumerate(columns):
        col_lower = col.lower()
        details = {"column": col, "index": i, "reasons": []}
        
        # 检查是否需要术语标准化
        for keyword in MEDICAL_COLUMN_KEYWORDS['standardization']:
            if keyword.lower() in col_lower:
                result["standardization_candidates"].append({
                    "column": col,
                    "index": i,
                    "matched_keyword": keyword
                })
                details["reasons"].append(f"术语标准化: 匹配关键词 '{keyword}'")
                break
        
        # 检查是否需要量纲统一
        for keyword in MEDICAL_COLUMN_KEYWORDS['unit_normalization']:
            if keyword.lower() in col_lower:
                result["unit_normalization_candidates"].append({
                    "column": col,
                    "index": i,
                    "matched_keyword": keyword
                })
                details["reasons"].append(f"量纲统一: 匹配关键词 '{keyword}'")
                break
        
        # 检查是否是单位列
        for keyword in MEDICAL_COLUMN_KEYWORDS['unit_columns']:
            if keyword.lower() in col_lower:
                result["unit_columns"].append({
                    "column": col,
                    "index": i,
                    "matched_keyword": keyword
                })
                details["reasons"].append(f"单位列: 匹配关键词 '{keyword}'")
                break
        
        if details["reasons"]:
            result["analysis_details"].append(details)
    
    # 如果有样本数据，进一步分析
    if sample_data and len(sample_data) > 0:
        result["sample_analysis"] = _analyze_sample_data(columns, sample_data)
    
    return result


def _analyze_sample_data(
    columns: List[str], 
    sample_data: List[List[str]]
) -> Dict[str, Any]:
    """
    分析样本数据，推断列的数据类型和特征
    
    Args:
        columns: 列名列表
        sample_data: 样本数据
        
    Returns:
        样本分析结果
    """
    analysis = {
        "numeric_columns": [],
        "text_columns": [],
        "mixed_columns": [],
        "date_columns": []
    }
    
    for i, col in enumerate(columns):
        values = [row[i] for row in sample_data if i < len(row)]
        
        # 统计数值型值的数量
        numeric_count = 0
        for v in values:
            try:
                float(v)
                numeric_count += 1
            except:
                pass
        
        if len(values) == 0:
            continue
            
        ratio = numeric_count / len(values)
        
        if ratio > 0.8:
            analysis["numeric_columns"].append(col)
        elif ratio < 0.2:
            analysis["text_columns"].append(col)
        else:
            analysis["mixed_columns"].append(col)
    
    return analysis


def get_csv_info_for_agent(file_path: str) -> str:
    """
    获取CSV文件信息，格式化为Agent可理解的字符串
    
    这是供Agent调用的主要函数，返回一个详细的分析报告。
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        格式化的CSV分析报告字符串
    """
    # 读取样本数据
    sample_result = read_csv_sample(file_path, sample_rows=3)
    
    if not sample_result["success"]:
        return f"读取CSV文件失败: {sample_result['error']}"
    
    # 分析列
    analysis = analyze_columns_for_standardization(
        sample_result["columns"],
        sample_result["sample_data"]
    )
    
    # 格式化输出
    lines = [
        f"=== CSV文件分析报告 ===",
        f"文件路径: {file_path}",
        f"总行数: 约 {sample_result['total_rows']} 行",
        f"列数: {len(sample_result['columns'])}",
        f"",
        f"--- 列名列表 ---",
    ]
    
    for i, col in enumerate(sample_result["columns"]):
        lines.append(f"  [{i}] {col}")
    
    lines.extend([
        f"",
        f"--- 样本数据（前3行）---",
    ])
    
    for i, row in enumerate(sample_result["sample_data"][:3]):
        # 将行数据与列名配对显示
        row_display = []
        for j, (col, val) in enumerate(zip(sample_result["columns"], row)):
            if len(val) > 30:
                val = val[:27] + "..."
            row_display.append(f"{col}: {val}")
        lines.append(f"  行{i+1}: {' | '.join(row_display[:5])}...")
    
    lines.extend([
        f"",
        f"--- 标准化分析建议 ---",
    ])
    
    if analysis["standardization_candidates"]:
        lines.append(f"建议进行术语标准化的列:")
        for item in analysis["standardization_candidates"]:
            lines.append(f"  - {item['column']} (匹配: {item['matched_keyword']})")
    else:
        lines.append(f"  未发现明显需要术语标准化的列")
    
    if analysis["unit_normalization_candidates"]:
        lines.append(f"建议进行量纲统一的列:")
        for item in analysis["unit_normalization_candidates"]:
            lines.append(f"  - {item['column']} (匹配: {item['matched_keyword']})")
    else:
        lines.append(f"  未发现明显需要量纲统一的列")
    
    if analysis["unit_columns"]:
        lines.append(f"单位相关列:")
        for item in analysis["unit_columns"]:
            lines.append(f"  - {item['column']}")
    
    return "\n".join(lines)


def read_csv_data(
    file_path: str,
    columns_to_process: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    encoding: str = 'utf-8'
) -> Dict[str, Any]:
    """
    读取CSV数据，返回结构化的数据
    
    Args:
        file_path: CSV文件路径
        columns_to_process: 要处理的列名列表，None表示所有列
        max_rows: 最大读取行数，None表示全部
        encoding: 文件编码
        
    Returns:
        包含CSV数据的字典:
        - success: 是否成功
        - columns: 列名列表
        - data: 数据列表（每行为一个字典）
        - row_count: 行数
        - error: 错误信息
    """
    result = {
        "success": False,
        "columns": [],
        "data": [],
        "row_count": 0,
        "error": None
    }
    
    if not os.path.exists(file_path):
        result["error"] = f"文件不存在: {file_path}"
        return result
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            result["columns"] = reader.fieldnames or []
            
            # 确定要处理的列
            if columns_to_process:
                cols_to_read = [c for c in columns_to_process if c in result["columns"]]
            else:
                cols_to_read = result["columns"]
            
            # 读取数据
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                
                # 只保留需要的列
                filtered_row = {k: row[k] for k in cols_to_read if k in row}
                result["data"].append(filtered_row)
            
            result["row_count"] = len(result["data"])
            result["success"] = True
            
    except UnicodeDecodeError:
        for enc in ['gbk', 'gb2312', 'latin-1']:
            try:
                return read_csv_data(file_path, columns_to_process, max_rows, enc)
            except:
                continue
        result["error"] = f"无法解析文件编码"
    except Exception as e:
        result["error"] = f"读取文件失败: {str(e)}"
    
    return result
