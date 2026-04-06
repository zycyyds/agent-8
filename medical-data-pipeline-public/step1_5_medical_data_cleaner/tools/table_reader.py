# -*- coding: utf-8 -*-
"""
表格数据读取工具

读取 table 文件夹中的 Excel 文件，按病人ID索引数据，
用于将图像/OCR提取的信息补充填充到结构化表格中。
"""
import os
import glob
from typing import Dict, Any, Optional, List


EXCEL_EXTENSIONS = {'.xlsx', '.xls'}


def find_table_file(data_dir: str) -> Optional[str]:
    """
    在 data_dir/table/ 下递归查找第一个 Excel 文件

    Args:
        data_dir: 数据根目录（包含病人ID文件夹和 table 文件夹）

    Returns:
        Excel 文件路径，未找到返回 None
    """
    table_dir = os.path.join(data_dir, 'table')
    if not os.path.isdir(table_dir):
        return None
    for ext in EXCEL_EXTENSIONS:
        files = glob.glob(os.path.join(table_dir, '**', f'*{ext}'), recursive=True)
        if files:
            return files[0]
    return None


def load_table_data(xlsx_path: str) -> Dict[str, Dict[str, Any]]:
    """
    加载 Excel 文件，返回以病人ID为键的字典

    Args:
        xlsx_path: Excel 文件路径

    Returns:
        {patient_id: {column_name: value, ...}, ...}

    Raises:
        ImportError: 缺少 pandas/openpyxl 依赖
        RuntimeError: 文件读取失败
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需要安装 pandas 和 openpyxl: pip install pandas openpyxl")

    try:
        df = pd.read_excel(xlsx_path, dtype=str)
    except Exception as e:
        raise RuntimeError(f"读取 Excel 文件失败: {e}")

    # 找 id 列（不区分大小写）
    id_col = None
    for col in df.columns:
        if str(col).strip().lower() == 'id':
            id_col = col
            break
    if id_col is None:
        id_col = df.columns[0]

    result = {}
    for _, row in df.iterrows():
        pid = str(row[id_col]).strip()
        if not pid or pid in ('nan', 'None'):
            continue
        row_dict = {
            str(col): str(val)
            for col, val in row.items()
            if str(val) not in ('nan', 'None', '') and val is not None
        }
        result[pid] = row_dict

    return result


def get_patient_table_data(
    table_data: Dict[str, Dict[str, Any]],
    patient_id: str
) -> Dict[str, Any]:
    """
    获取指定病人的表格数据，支持多种ID匹配方式

    Args:
        table_data: load_table_data 返回的数据
        patient_id: 病人ID（文件夹名）

    Returns:
        该病人的列数据字典，未找到返回空字典
    """
    if not table_data:
        return {}

    # 1. 精确匹配
    if patient_id in table_data:
        return table_data[patient_id]

    # 2. 数字归一化匹配（去除前导零）
    pid_norm = patient_id.lstrip('0')
    for key in table_data:
        if key.lstrip('0') == pid_norm:
            return table_data[key]

    return {}


def get_table_columns(table_data: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    获取表格所有列名（排除 id 列），保持首次出现顺序

    Args:
        table_data: load_table_data 返回的数据

    Returns:
        列名列表
    """
    all_cols: List[str] = []
    seen: set = set()
    for row in table_data.values():
        for col in row:
            if col.lower() != 'id' and col not in seen:
                seen.add(col)
                all_cols.append(col)
    return all_cols
