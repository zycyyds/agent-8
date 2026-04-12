# -*- coding: utf-8 -*-
"""
将旧格式的 CSV 结果文件转换为新格式：
1. {Category}_原始 → {Category}_抽取（键值对格式 "名称:值"）
2. {Category}_标准化 + {Category}_量纲统一 → {Category}_标准化（合并）
3. 去掉 {Category}_量纲统一 列
"""
import csv
import os
import sys
import re
from datetime import datetime


ENTITY_CATEGORIES = [
    "Test", "Disease", "Drug", "Symptom", "Treatment",
    "Anatomy", "LabValue", "Finding", "Other"
]


def convert_original_to_extraction(text: str) -> str:
    """
    将 _原始 格式转为 _抽取 格式
    
    原始: "类风湿因子 20.0 IU/mL; 超敏C反应蛋白 1.35 mg/L"
    抽取: "类风湿因子:20.0 IU/mL; 超敏C反应蛋白:1.35 mg/L"
    """
    if not text or not text.strip():
        return ""
    
    items = [item.strip() for item in text.split(";")]
    converted = []
    
    for item in items:
        if not item:
            continue
        # 尝试把 "名称 值 单位" 转为 "名称:值 单位"
        # 找到第一个空格后面跟数字/阴性/阳性等值的位置
        # 匹配模式：名称 + 空格 + 值（数字开头、阴性、阳性、未见、可见、未做等）
        match = re.match(
            r'^(.+?)\s+([-+]?\d[\d.]*.*|阴性.*|阳性.*|未见.*|可见.*|未做.*|正常.*|异常.*|双向.*)$',
            item
        )
        if match:
            name = match.group(1).strip()
            value = match.group(2).strip()
            converted.append(f"{name}:{value}")
        else:
            # 没有值的情况，保持原样
            converted.append(item)
    
    return "; ".join(converted)


def merge_standardization(std_text: str, unit_text: str) -> str:
    """
    合并 _标准化 和 _量纲统一 为一列
    """
    parts = []
    if std_text and std_text.strip():
        parts.append(std_text.strip())
    if unit_text and unit_text.strip():
        parts.append(unit_text.strip())
    return "; ".join(parts)


def convert_csv(input_path: str, output_path: str = None) -> str:
    """
    转换 CSV 文件格式
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（默认在同目录生成 _converted 文件）
    
    Returns:
        输出文件路径
    """
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return ""
    
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_converted{ext}"
    
    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        original_columns = reader.fieldnames
        rows = list(reader)
    
    print(f"📂 读取文件: {input_path}")
    print(f"   - 行数: {len(rows)}")
    print(f"   - 列数: {len(original_columns)}")
    
    # 检测需要转换的列
    original_cols = {}   # {Category}_原始
    std_cols = {}        # {Category}_标准化
    unit_cols = {}       # {Category}_量纲统一
    
    for col in original_columns:
        for cat in ENTITY_CATEGORIES:
            if col == f"{cat}_原始":
                original_cols[cat] = col
            elif col == f"{cat}_标准化":
                std_cols[cat] = col
            elif col == f"{cat}_量纲统一":
                unit_cols[cat] = col
    
    has_original = bool(original_cols)
    has_unit = bool(unit_cols)
    
    if not has_original and not has_unit:
        print("⚠️  文件已是新格式（没有 _原始 或 _量纲统一 列），无需转换")
        return input_path
    
    print(f"\n🔄 转换中...")
    if has_original:
        print(f"   - 发现 {len(original_cols)} 个 _原始 列 → 转为 _抽取")
    if has_unit:
        print(f"   - 发现 {len(unit_cols)} 个 _量纲统一 列 → 合并到 _标准化")
    
    # 构建新列名列表
    new_columns = []
    skip_columns = set()
    
    for col in original_columns:
        # 跳过 _量纲统一 列（已合并到 _标准化）
        is_unit_col = False
        for cat in ENTITY_CATEGORIES:
            if col == f"{cat}_量纲统一":
                is_unit_col = True
                skip_columns.add(col)
                break
        
        if is_unit_col:
            continue
        
        # _原始 → _抽取
        renamed = False
        for cat in ENTITY_CATEGORIES:
            if col == f"{cat}_原始":
                new_columns.append(f"{cat}_抽取")
                renamed = True
                break
        
        if not renamed:
            new_columns.append(col)
    
    # 转换每一行数据
    new_rows = []
    for row in rows:
        new_row = {}
        
        for col in original_columns:
            if col in skip_columns:
                continue
            
            value = row.get(col, "")
            
            # 检查是否是 _原始 列
            is_original = False
            for cat in ENTITY_CATEGORIES:
                if col == f"{cat}_原始":
                    # 转换为 _抽取 格式
                    new_row[f"{cat}_抽取"] = convert_original_to_extraction(value)
                    is_original = True
                    break
            
            if is_original:
                continue
            
            # 检查是否是 _标准化 列（需要合并 _量纲统一）
            is_std = False
            for cat in ENTITY_CATEGORIES:
                if col == f"{cat}_标准化":
                    unit_value = row.get(f"{cat}_量纲统一", "")
                    new_row[col] = merge_standardization(value, unit_value)
                    is_std = True
                    break
            
            if is_std:
                continue
            
            # 其他列直接复制
            new_row[col] = value
        
        new_rows.append(new_row)
    
    # 写入新文件
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_columns, extrasaction='ignore')
        writer.writeheader()
        for row in new_rows:
            complete_row = {col: row.get(col, "") for col in new_columns}
            writer.writerow(complete_row)
    
    print(f"\n✅ 转换完成: {output_path}")
    print(f"   - 行数: {len(new_rows)}")
    print(f"   - 列数: {len(new_columns)} (原 {len(original_columns)})")
    
    # 显示列变化
    removed = len(original_columns) - len(new_columns)
    if removed > 0:
        print(f"   - 移除 {removed} 个 _量纲统一 列")
    renamed_count = len(original_cols)
    if renamed_count > 0:
        print(f"   - 重命名 {renamed_count} 个 _原始 → _抽取")
    
    return output_path


def main():
    if len(sys.argv) < 2:
        print("用法: python convert_csv_format.py <csv文件路径> [输出路径]")
        print()
        print("示例:")
        print("  python convert_csv_format.py results/patients_results_20260206_151627.csv")
        print("  python convert_csv_format.py input.csv output.csv")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_csv(input_path, output_path)


if __name__ == "__main__":
    main()
