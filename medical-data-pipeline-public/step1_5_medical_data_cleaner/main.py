# -*- coding: utf-8 -*-
"""
Medical Data Cleaner - 医学数据清洗Agent框架

主程序入口，支持处理多种数据格式：
1. 图片文件 - 通过OCR提取文本后进行信息抽取和标准化
2. 文本文件 - 进行预处理、信息抽取和标准化
3. CSV结构化数据 - 跳过预处理和信息抽取，直接进行标准化和量纲统一

使用方法:
    # 自动检测数据类型并处理
    python main.py --input /path/to/data
    
    # 处理CSV文件
    python main.py --input /path/to/data.csv --type csv
    
    # 批量处理目录
    python main.py --input /path/to/directory --batch
    
    # 查询模式
    python main.py --query
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

# 设置导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

src_path = os.path.abspath(os.path.join(parent_dir, "../../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from agents.unified_processing_agent import UnifiedProcessingAgent, process_medical_data
from agents.data_type_detector_agent import DataTypeDetectorAgent
from tools.data_type_detector import detect_data_type
from tools.csv_reader import get_csv_info_for_agent
from config.settings import DataType, get_api_config


def simplify_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    简化处理结果，只保留标准化前后的对比信息
    
    Args:
        result: 完整的处理结果
        
    Returns:
        简化后的结果，只包含标准化对比
    """
    if not result.get("success") or "data" not in result:
        return result
    
    simplified = {
        "success": result.get("success"),
        "file_path": result.get("file_path"),
        "processed_at": result.get("processed_at"),
        "data_type": result.get("data_type"),
        "statistics": result.get("statistics"),
        "standardization_mappings": []
    }
    
    # 提取标准化映射
    data = result.get("data", [])
    
    # 获取需要标准化的列
    std_columns = result.get("llm_analysis", {}).get("standardization_columns", [])
    value_columns = result.get("llm_analysis", {}).get("value_columns", [])
    
    # 用于去重
    seen_mappings = set()
    
    for row in data:
        # 提取术语标准化映射
        for col in std_columns:
            if col in row and row[col]:
                original = row[col]
                std_code = row.get(f"{col}_standard_code")
                std_name = row.get(f"{col}_standard_name")
                std_system = row.get(f"{col}_standard_system")
                
                if std_code:
                    # 创建唯一键用于去重
                    mapping_key = f"{col}|{original}|{std_code}"
                    if mapping_key not in seen_mappings:
                        seen_mappings.add(mapping_key)
                        simplified["standardization_mappings"].append({
                            "column": col,
                            "original": original,
                            "standard_code": std_code,
                            "standard_name": std_name,
                            "standard_system": std_system
                        })
        
        # 提取量纲统一映射
        for col in value_columns:
            if col in row and row[col]:
                original_value = row[col]
                normalized_value = row.get(f"{col}_normalized")
                normalized_unit = row.get(f"{col}_normalized_unit")
                
                if normalized_value is not None:
                    mapping_key = f"{col}|{original_value}|{normalized_value}"
                    if mapping_key not in seen_mappings:
                        seen_mappings.add(mapping_key)
                        simplified["standardization_mappings"].append({
                            "column": col,
                            "type": "unit_normalization",
                            "original_value": original_value,
                            "normalized_value": normalized_value,
                            "normalized_unit": normalized_unit
                        })
    
    # 按列名和原始值排序
    simplified["standardization_mappings"].sort(
        key=lambda x: (x.get("column", ""), x.get("original", "") or str(x.get("original_value", "")))
    )
    
    return simplified


def save_patient_results(
    result: Dict[str, Any],
    output_dir: str,
    compact: bool = False,
    verbose: bool = True
) -> Dict[str, str]:
    """
    按病人分别保存处理结果
    
    Args:
        result: 目录处理结果
        output_dir: 输出目录
        compact: 是否简洁输出
        verbose: 是否详细输出
        
    Returns:
        保存的文件路径 {patient_id: file_path}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files = {}
    
    # 创建病人结果目录
    patients_dir = os.path.join(output_dir, f"patients_{timestamp}")
    os.makedirs(patients_dir, exist_ok=True)
    
    patients = result.get("patients", {})
    
    for patient_id, patient_data in patients.items():
        # 准备单个病人的输出数据
        patient_output = {
            "patient_id": patient_id,
            "processed_at": result.get("processed_at"),
            "root_path": result.get("root_path"),
            "categories": patient_data.get("categories", []),
            "statistics": {
                "file_count": patient_data.get("file_count", 0),
                "processed_count": patient_data.get("processed_count", 0),
                "failed_count": patient_data.get("failed_count", 0)
            },
            "files": patient_data.get("files", []),
            "merged_data": patient_data.get("merged_data", {})
        }
        
        # 如果使用简洁模式，简化输出
        if compact:
            patient_output = simplify_patient_result(patient_output)
        
        # 保存文件
        suffix = "_compact" if compact else "_full"
        output_file = os.path.join(patients_dir, f"patient_{patient_id}{suffix}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(patient_output, f, indent=2, ensure_ascii=False, default=str)
        
        output_files[patient_id] = output_file
        
        if verbose:
            print(f"✅ 病人 {patient_id} 结果已保存: {output_file}")
    
    # 保存汇总文件
    summary = {
        "processed_at": result.get("processed_at"),
        "root_path": result.get("root_path"),
        "statistics": result.get("statistics"),
        "patient_files": output_files,
        "patient_summary": {
            pid: {
                "categories": pdata.get("categories", []),
                "file_count": pdata.get("file_count", 0),
                "processed_count": pdata.get("processed_count", 0),
                "failed_count": pdata.get("failed_count", 0)
            }
            for pid, pdata in patients.items()
        }
    }
    
    summary_file = os.path.join(patients_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    if verbose:
        print(f"\n📋 汇总文件已保存: {summary_file}")
    
    output_files["_summary"] = summary_file
    return output_files


def simplify_patient_result(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    简化单个病人的处理结果
    
    Args:
        patient_data: 病人处理数据
        
    Returns:
        简化后的数据
    """
    simplified = {
        "patient_id": patient_data.get("patient_id"),
        "processed_at": patient_data.get("processed_at"),
        "categories": patient_data.get("categories", []),
        "statistics": patient_data.get("statistics"),
        "files_processed": [
            {
                "filename": f.get("filename"),
                "category": f.get("category"),
                "success": f.get("success")
            }
            for f in patient_data.get("files", [])
        ]
    }
    
    # 提取合并的实体和标准化结果
    merged = patient_data.get("merged_data", {})
    
    if merged.get("entities"):
        simplified["entities"] = merged["entities"]
    
    if merged.get("standardized_terms"):
        # 去重标准化结果
        seen = set()
        unique_terms = []
        for term in merged["standardized_terms"]:
            key = f"{term.get('original')}|{term.get('standard_code')}"
            if key not in seen:
                seen.add(key)
                unique_terms.append(term)
        simplified["standardized_terms"] = unique_terms
    
    return simplified


def _parse_value_and_unit(text: str):
    """
    从文本中分离数值和单位
    
    Args:
        text: 包含数值和单位的文本，如 "10 度/秒" 或 "未见眼震"
        
    Returns:
        (数值, 单位) 元组
    """
    import re
    
    if not text:
        return "", ""
    
    text = str(text).strip()
    
    # 尝试匹配数值+单位格式
    patterns = [
        r'^([-+]?\d*\.?\d+)\s*(.*)$',
        r'^([<>≤≥]?\s*[-+]?\d*\.?\d+)\s*(.*)$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            value_part = match.group(1).strip()
            unit_part = match.group(2).strip()
            if value_part:
                return value_part, unit_part
    
    return text, ""


def _build_patient_rows(
    patient_id: str,
    patient_data: Dict[str, Any],
    entity_keys: List,
    ENTITY_CATEGORIES: List[str],
    table_columns: List[str] = None
) -> List[Dict]:
    """
    为单个病人构建所有行数据
    
    Args:
        patient_id: 病人ID
        patient_data: 病人数据
        entity_keys: 该病人的实体展开列 [(cat, clean_name, original_name), ...]
        ENTITY_CATEGORIES: 实体类别列表
    
    Returns:
        该病人的所有行数据
    """
    rows = []
    
    for file_info in patient_data.get("files", []):
        row = {
            "patient_id": patient_id,
            "folder_category": file_info.get("category", ""),
            "filename": file_info.get("filename", ""),
            "success": file_info.get("success", False),
        }
        
        # OCR结果
        ocr_text = file_info.get("ocr_text", "")
        row["ocr_text"] = ocr_text.replace("\n", " ").replace("\r", "")[:3000] if ocr_text else ""
        
        # 预处理结果
        preprocessed_text = file_info.get("preprocessed_text", "")
        row["preprocessed_text"] = preprocessed_text.replace("\n", " ").replace("\r", "")[:3000] if preprocessed_text else ""
        
        # 处理信息抽取和标准化结果
        extraction = file_info.get("extraction_result", [])
        standardized = file_info.get("standardized_result", [])
        
        # 创建标准化结果的索引
        std_by_name = {}
        for entity in standardized:
            name = entity.get("name") or entity.get("original_text", "")
            if name:
                std_by_name[name] = entity
        
        # 按类别汇总
        by_category = {cat: {"抽取": [], "标准化": []} for cat in ENTITY_CATEGORIES}
        entity_dict = {}
        
        for entity in extraction:
            cat = entity.get("category", "Other")
            if cat not in by_category:
                cat = "Other"
            
            name = (entity.get("name") or entity.get("original_text", "")).strip()
            value = entity.get("value")
            unit = entity.get("unit")
            
            if name:
                clean_name = name.replace("/", "-").replace("\\", "-").replace(":", "-")
                std_entity = std_by_name.get(name, {})
                entity_dict[(cat, clean_name)] = (entity, std_entity)
            
            # 抽取信息
            if name:
                if value is not None:
                    entity_str = f"{name}:{value}"
                    if unit:
                        entity_str += f" {unit}"
                else:
                    entity_str = name
                by_category[cat]["抽取"].append(entity_str)
            
            # 标准化信息
            std_entity = std_by_name.get(name, {})
            std_parts = []
            
            std_name = std_entity.get("standard_name")
            std_code = std_entity.get("standard_code")
            std_system = std_entity.get("standard_system")
            
            if std_name and std_name != name:
                std_str = f"{name}→{std_name}"
                if std_system:
                    std_str += f"[{std_system}]"
                std_parts.append(std_str)
            
            norm_value = std_entity.get("normalized_value")
            norm_unit = std_entity.get("normalized_unit")
            
            value_changed = (norm_value is not None and value is not None and 
                            str(norm_value) != str(value))
            unit_changed = (norm_unit is not None and unit is not None and 
                           norm_unit != unit)
            
            if value_changed or unit_changed:
                original_str = f"{value}" + (f" {unit}" if unit else "")
                normalized_str = f"{norm_value}" + (f" {norm_unit}" if norm_unit else "")
                std_parts.append(f"{name}: {original_str}→{normalized_str}")
            
            if std_parts:
                by_category[cat]["标准化"].append("; ".join(std_parts))
        
        for cat in ENTITY_CATEGORIES:
            row[f"{cat}_抽取"] = "; ".join(by_category[cat]["抽取"]) if by_category[cat]["抽取"] else ""
            row[f"{cat}_标准化"] = "; ".join(by_category[cat]["标准化"]) if by_category[cat]["标准化"] else ""
        
        # 按实体名称展开为独立列
        for cat, clean_name, original_name in entity_keys:
            col_prefix = f"Extracted_{cat}_{clean_name}"
            entity_pair = entity_dict.get((cat, clean_name))
            
            if entity_pair:
                entity, std_entity = entity_pair
                value = entity.get("value")
                unit = entity.get("unit")
                
                if value is not None:
                    entity_str = str(value)
                    if unit:
                        entity_str += f" {unit}"
                else:
                    entity_str = ""
                
                parsed_value, parsed_unit = _parse_value_and_unit(entity_str)
                row[col_prefix] = entity_str
                row[f"{col_prefix}_value"] = parsed_value
                row[f"{col_prefix}_unit"] = parsed_unit
            else:
                row[col_prefix] = ""
                row[f"{col_prefix}_value"] = ""
                row[f"{col_prefix}_unit"] = ""
        
        # 时间信息
        temporal_info = file_info.get("temporal_info", [])
        temporal_strs = []
        if temporal_info:
            for t in temporal_info:
                event = t.get("event", "")
                time_expr = t.get("time_expression", "")
                norm_time = t.get("normalized_time", "")
                if event and time_expr:
                    if norm_time and norm_time != time_expr:
                        temporal_strs.append(f"{event}: {time_expr} → {norm_time}")
                    else:
                        temporal_strs.append(f"{event}: {time_expr}")
        row["temporal_info"] = "; ".join(temporal_strs) if temporal_strs else ""
        
        # 剂量信息
        quantity_info = file_info.get("quantity_info", [])
        qty_strs = []
        if quantity_info:
            for q in quantity_info:
                drug = q.get("drug_or_treatment", "")
                amount = q.get("amount", "")
                freq = q.get("frequency", "")
                duration = q.get("duration", "")
                parts = [p for p in [drug, amount, freq, duration] if p]
                if parts:
                    qty_strs.append(" ".join(parts))
        row["quantity_info"] = "; ".join(qty_strs) if qty_strs else ""
        
        # 关系信息
        relations = file_info.get("relations", [])
        rel_strs = []
        if relations:
            for r in relations:
                source = r.get("source", "")
                target = r.get("target", "")
                rel_type = r.get("relation_type", "")
                if source and target:
                    rel_strs.append(f"{source} --{rel_type}--> {target}")
        row["relations"] = "; ".join(rel_strs) if rel_strs else ""
        
        row["error"] = file_info.get("error", "")
        row["file_path"] = file_info.get("path", "")

        # 表格数据（每行都填入该病人的表格字段）
        if table_columns:
            patient_table = patient_data.get("table_data", {})
            for col in table_columns:
                row[f"Table_{col}"] = patient_table.get(col, "")

        rows.append(row)
    
    return rows


def _collect_entity_keys_for_patient(
    patient_data: Dict[str, Any],
    ENTITY_CATEGORIES: List[str]
) -> List:
    """收集单个病人的所有实体名称"""
    entity_keys = set()
    
    for file_info in patient_data.get("files", []):
        extraction = file_info.get("extraction_result", [])
        standardized = file_info.get("standardized_result", [])
        
        for entity_list in [extraction, standardized]:
            for entity in entity_list:
                cat = entity.get("category", "Other")
                if cat not in ENTITY_CATEGORIES:
                    cat = "Other"
                name = (entity.get("name") or entity.get("original_text", "")).strip()
                if name:
                    clean_name = name.replace("/", "-").replace("\\", "-").replace(":", "-")
                    entity_keys.add((cat, clean_name, name))
    
    return sorted(entity_keys, key=lambda x: (
        ENTITY_CATEGORIES.index(x[0]) if x[0] in ENTITY_CATEGORIES else 999, x[1]
    ))


def _build_columns(
    entity_keys: List,
    ENTITY_CATEGORIES: List[str],
    table_columns: List[str] = None
) -> List[str]:
    """构建列顺序"""
    priority_columns = ["patient_id", "folder_category", "filename", "success", "ocr_text", "preprocessed_text"]

    category_columns = []
    for cat in ENTITY_CATEGORIES:
        for suffix in ["_抽取", "_标准化"]:
            category_columns.append(f"{cat}{suffix}")

    entity_columns = []
    for cat, clean_name, original_name in entity_keys:
        col_prefix = f"Extracted_{cat}_{clean_name}"
        entity_columns.append(col_prefix)
        entity_columns.append(f"{col_prefix}_value")
        entity_columns.append(f"{col_prefix}_unit")

    info_columns = ["temporal_info", "quantity_info", "relations", "error", "file_path"]

    # 表格列（以 Table_ 为前缀，排除 id 列）
    table_col_list = [f"Table_{col}" for col in (table_columns or []) if col.lower() != 'id']

    columns = priority_columns + category_columns + entity_columns + info_columns + table_col_list

    # 去重
    seen = set()
    unique_columns = []
    for col in columns:
        if col not in seen:
            seen.add(col)
            unique_columns.append(col)
    return unique_columns


def _write_csv(rows: List[Dict], columns: List[str], csv_path: str) -> None:
    """写入CSV文件"""
    import csv
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            complete_row = {col: row.get(col, "") for col in columns}
            writer.writerow(complete_row)


def save_patient_results_as_csv(
    result: Dict[str, Any],
    output_dir: str,
    verbose: bool = True
) -> str:
    """
    将病人处理结果保存为CSV格式
    
    每个病人单独保存一个CSV文件，放在以时间戳命名的新文件夹中。
    同时生成一个包含所有病人的汇总CSV。
    
    输出结构：
        output_dir/
          results_YYYYMMDD_HHMMSS/
            patient_36907.csv
            patient_36909.csv
            ...
            all_patients.csv  (汇总)
    
    Args:
        result: 目录处理结果
        output_dir: 输出根目录
        verbose: 是否详细输出
        
    Returns:
        结果文件夹路径
    """
    ENTITY_CATEGORIES = ["Test", "Disease", "Drug", "Symptom", "Treatment", "Anatomy", "LabValue", "Finding", "Other"]

    patients = result.get("patients", {})
    if not patients:
        return ""

    # 收集所有表格列名（保持首次出现顺序，排除 id 列）
    all_table_columns: List[str] = []
    seen_table_cols: set = set()
    for pid in sorted(patients.keys()):
        for col in patients[pid].get("table_data", {}):
            if col.lower() != 'id' and col not in seen_table_cols:
                seen_table_cols.add(col)
                all_table_columns.append(col)

    if verbose and all_table_columns:
        print(f"\n📋 检测到表格数据: {len(all_table_columns)} 列将写入 CSV")

    # 创建带时间戳的结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join(output_dir, f"results_{timestamp}")
    os.makedirs(results_folder, exist_ok=True)

    if verbose:
        print(f"\n📁 结果保存目录: {results_folder}")

    # ========== 1. 每个病人单独保存一个CSV ==========
    all_rows = []           # 汇总用
    all_entity_keys = set() # 汇总用
    patient_files = {}

    for patient_id in sorted(patients.keys()):
        patient_data = patients[patient_id]

        # 收集该病人的实体名称
        patient_entity_keys = _collect_entity_keys_for_patient(patient_data, ENTITY_CATEGORIES)

        # 构建该病人的行数据（含表格列）
        patient_rows = _build_patient_rows(
            patient_id, patient_data, patient_entity_keys, ENTITY_CATEGORIES,
            table_columns=all_table_columns
        )

        if not patient_rows:
            continue

        # 构建该病人的列（含表格列）
        patient_columns = _build_columns(patient_entity_keys, ENTITY_CATEGORIES,
                                         table_columns=all_table_columns)

        # 写入该病人的CSV
        patient_csv = os.path.join(results_folder, f"patient_{patient_id}.csv")
        _write_csv(patient_rows, patient_columns, patient_csv)

        patient_files[patient_id] = patient_csv

        if verbose:
            print(f"  ✅ 病人 {patient_id}: {len(patient_rows)} 行, "
                  f"{len(patient_entity_keys)} 个实体 → {os.path.basename(patient_csv)}")

        # 收集到汇总
        for key in patient_entity_keys:
            all_entity_keys.add(key)
        all_rows.extend(patient_rows)

    # ========== 2. 保存汇总CSV（所有病人） ==========
    if all_rows:
        sorted_all_keys = sorted(all_entity_keys, key=lambda x: (
            ENTITY_CATEGORIES.index(x[0]) if x[0] in ENTITY_CATEGORIES else 999, x[1]
        ))
        all_columns = _build_columns(sorted_all_keys, ENTITY_CATEGORIES,
                                     table_columns=all_table_columns)

        # 重新构建汇总行（使用全局列，含表格列）
        summary_rows = []
        for patient_id in sorted(patients.keys()):
            patient_data = patients[patient_id]
            rows = _build_patient_rows(
                patient_id, patient_data, sorted_all_keys, ENTITY_CATEGORIES,
                table_columns=all_table_columns
            )
            summary_rows.extend(rows)

        all_csv = os.path.join(results_folder, "all_patients.csv")
        _write_csv(summary_rows, all_columns, all_csv)
        
        if verbose:
            print(f"\n  📊 汇总文件: all_patients.csv")
            print(f"     - 总行数: {len(summary_rows)}")
            print(f"     - 列数: {len(all_columns)}")
            print(f"     - 病人数: {len(patient_files)}")
    
    if verbose:
        print(f"\n✅ 所有结果已保存到: {results_folder}")
    
    return results_folder


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║         Medical Data Cleaner - 医学数据清洗Agent框架              ║
║                                                                   ║
║  支持的数据类型:                                                  ║
║    • 图片文件 (jpg, png, etc.) - OCR -> 预处理 -> 抽取 -> 标准化  ║
║    • 文本文件 (txt, md, etc.)  - 预处理 -> 抽取 -> 标准化         ║
║    • CSV文件                   - 直接标准化和量纲统一             ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


async def detect_and_analyze(input_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    检测并分析输入数据
    
    Args:
        input_path: 输入路径
        verbose: 是否详细输出
        
    Returns:
        分析结果
    """
    if verbose:
        print(f"\n📂 分析输入: {input_path}")
        print("-" * 60)
    
    # 基础类型检测
    data_type, metadata = detect_data_type(input_path)
    
    if verbose:
        print(f"检测到类型: {data_type.value}")
        print(f"处理流程: {' -> '.join(metadata.get('processing_stages', []))}")
    
    result = {
        "input_path": input_path,
        "data_type": data_type.value,
        "metadata": metadata
    }
    
    # 对CSV进行详细分析
    if data_type == DataType.CSV:
        csv_info = get_csv_info_for_agent(input_path)
        result["csv_analysis"] = csv_info
        if verbose:
            print(f"\n{csv_info}")
    
    return result


async def process_single_input(
    input_path: str,
    agent: UnifiedProcessingAgent,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    auto_save: bool = True,
    compact: bool = False
) -> Dict[str, Any]:
    """
    处理单个输入
    
    Args:
        input_path: 输入路径
        agent: 处理Agent
        output_dir: 输出目录（如果为None且auto_save=True，会自动保存到默认目录）
        verbose: 是否详细输出
        auto_save: 是否自动保存结果
        compact: 是否简洁输出（仅保留标准化对比）
        
    Returns:
        处理结果
    """
    from agentscope.message import Msg
    
    if verbose:
        print(f"\n🔄 处理中: {input_path}")
        print("-" * 60)
    
    # 调用Agent处理
    msg = Msg(name="User", content=input_path, role="user")
    result_msg = await agent.reply(msg)
    
    result = result_msg.metadata if result_msg.metadata else {}
    
    # 如果使用简洁模式，简化结果
    if compact and result.get("success"):
        result = simplify_result(result)
    
    # 自动保存结果
    should_save = (output_dir is not None) or (auto_save and result.get("success"))
    
    if should_save and result.get("success"):
        # 如果没有指定输出目录，使用默认目录
        if output_dir is None:
            output_dir = os.path.join(current_dir, "results")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查是否是目录处理结果（按病人分组保存）
        if result.get("data_type") == "directory" and result.get("patients"):
            result["output_files"] = save_patient_results(
                result, output_dir, compact, verbose
            )
            # 保存CSV格式（每个病人单独一个CSV + 汇总CSV）
            results_folder = save_patient_results_as_csv(result, output_dir, verbose)
            if results_folder:
                result["csv_results_folder"] = results_folder
        else:
            # 普通文件处理结果
            base_name = os.path.basename(input_path)
            name_without_ext = os.path.splitext(base_name)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_compact" if compact else "_processed"
            output_file = os.path.join(output_dir, f"{name_without_ext}{suffix}_{timestamp}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            if verbose:
                print(f"✅ 结果已保存: {output_file}")
            
            result["output_file"] = output_file
    
    if verbose:
        print(f"处理状态: {'✅ 成功' if result.get('success') else '❌ 失败'}")
        if result.get("error"):
            print(f"错误信息: {result['error']}")
        
        # 目录处理结果
        if result.get("data_type") == "directory":
            stats = result.get("statistics", {})
            print(f"处理统计:")
            print(f"  - 总病人数: {stats.get('total_patients', 0)}")
            print(f"  - 总文件数: {stats.get('total_files', 0)}")
            print(f"  - 成功处理: {stats.get('processed_files', 0)}")
            print(f"  - 处理失败: {stats.get('failed_files', 0)}")
        elif result.get("statistics"):
            stats = result["statistics"]
            print(f"处理统计:")
            print(f"  - 总行数: {stats.get('total_rows', 'N/A')}")
            print(f"  - 术语标准化: {stats.get('terms_standardized', 0)} 成功")
            print(f"  - 量纲统一: {stats.get('values_normalized', 0)} 已转换")
        
        # 在简洁模式下显示映射数量
        if compact and result.get("standardization_mappings"):
            print(f"  - 标准化映射: {len(result['standardization_mappings'])} 条（去重后）")
    
    return result


async def batch_process(
    input_dir: str,
    agent: UnifiedProcessingAgent,
    output_dir: Optional[str] = None,
    file_pattern: Optional[str] = None,
    max_files: Optional[int] = None,
    verbose: bool = True,
    compact: bool = False
) -> Dict[str, Any]:
    """
    批量处理目录中的文件
    
    Args:
        input_dir: 输入目录
        agent: 处理Agent
        output_dir: 输出目录
        file_pattern: 文件匹配模式
        max_files: 最大处理文件数
        verbose: 是否详细输出
        compact: 是否简洁输出
        
    Returns:
        批量处理结果
    """
    import glob
    
    if not os.path.isdir(input_dir):
        return {"error": f"目录不存在: {input_dir}", "success": False}
    
    # 获取文件列表
    if file_pattern:
        files = glob.glob(os.path.join(input_dir, file_pattern))
    else:
        # 获取所有支持的文件
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.txt', '*.md', '*.csv']:
            files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if max_files:
        files = files[:max_files]
    
    if verbose:
        print(f"\n📁 批量处理: {input_dir}")
        print(f"找到 {len(files)} 个文件")
        print("=" * 60)
    
    results = {
        "input_dir": input_dir,
        "total_files": len(files),
        "processed": 0,
        "success": 0,
        "failed": 0,
        "file_results": []
    }
    
    for i, file_path in enumerate(files, 1):
        if verbose:
            print(f"\n[{i}/{len(files)}] ", end="")
        
        try:
            file_result = await process_single_input(
                file_path, agent, output_dir, verbose, compact=compact
            )
            results["file_results"].append({
                "file": file_path,
                "success": file_result.get("success", False),
                "data_type": file_result.get("data_type"),
                "error": file_result.get("error")
            })
            
            results["processed"] += 1
            if file_result.get("success"):
                results["success"] += 1
            else:
                results["failed"] += 1
                
        except Exception as e:
            results["failed"] += 1
            results["file_results"].append({
                "file": file_path,
                "success": False,
                "error": str(e)
            })
            if verbose:
                print(f"❌ 处理失败: {e}")
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"批量处理完成:")
        print(f"  - 总文件: {results['total_files']}")
        print(f"  - 成功: {results['success']}")
        print(f"  - 失败: {results['failed']}")
    
    return results


async def _handle_interactive_command(
    user_input: str,
    agent: UnifiedProcessingAgent,
    output_dir: Optional[str],
    analyze_mode: bool,
    output_func=print,
) -> dict:
    command = user_input.strip()
    if not command:
        return {
            "analyze_mode": analyze_mode,
            "should_quit": False,
            "handled": False,
            "event": "empty",
            "success": True,
        }

    lowered = command.lower()
    if lowered == "quit":
        output_func("👋 再见!")
        return {
            "analyze_mode": analyze_mode,
            "should_quit": True,
            "handled": True,
            "event": "quit",
            "success": True,
        }

    if lowered == "analyze":
        next_mode = not analyze_mode
        output_func(f"已切换到{'分析' if next_mode else '处理'}模式")
        return {
            "analyze_mode": next_mode,
            "should_quit": False,
            "handled": True,
            "event": "toggle_analyze",
            "success": True,
        }

    if lowered == "stats":
        stats = agent.get_stats()
        output_func("处理统计:")
        output_func(f"  - 总处理数: {stats['total_processed']}")
        output_func(f"  - 按类型: {stats['by_type']}")
        output_func(f"  - 错误数: {stats['errors']}")
        return {
            "analyze_mode": analyze_mode,
            "should_quit": False,
            "handled": True,
            "event": "stats",
            "success": True,
            "stats": stats,
        }

    if analyze_mode:
        analysis_result = await detect_and_analyze(command, verbose=True)
        return {
            "analyze_mode": analyze_mode,
            "should_quit": False,
            "handled": True,
            "event": "analyze_input",
            "success": bool((analysis_result or {}).get("success", True)),
            "input": command,
            "result": analysis_result,
        }

    process_result = await process_single_input(command, agent, output_dir, verbose=True)
    return {
        "analyze_mode": analyze_mode,
        "should_quit": False,
        "handled": True,
        "event": "process_input",
        "success": bool((process_result or {}).get("success", False)),
        "input": command,
        "result": process_result,
    }


async def run_interactive_session(
    agent: UnifiedProcessingAgent,
    output_dir: Optional[str] = None,
    input_func=input,
    output_func=print,
    show_banner: bool = True,
) -> Dict[str, Any]:
    if show_banner:
        output_func("\n📝 交互模式 - 输入文件路径进行处理")
        output_func("   输入 'quit' 退出, 'analyze' 切换到分析模式, 'stats' 查看统计")
        output_func("-" * 60)

    analyze_mode = False
    session_summary = {
        "quit_reason": "unknown",
        "processed_inputs": 0,
        "analyze_requests": 0,
        "stats_requests": 0,
        "success_count": 0,
        "failure_count": 0,
        "last_input": None,
        "last_event": None,
        "output_dir": output_dir,
        "session_events": [],
    }

    while True:
        try:
            prompt = "分析> " if analyze_mode else "处理> "
            user_input = input_func(f"\n{prompt}").strip()
            command_result = await _handle_interactive_command(
                user_input=user_input,
                agent=agent,
                output_dir=output_dir,
                analyze_mode=analyze_mode,
                output_func=output_func,
            )
            analyze_mode = command_result.get("analyze_mode", analyze_mode)
            session_summary["last_event"] = command_result.get("event")

            event = command_result.get("event")
            session_event = {
                "input": user_input,
                "event": event,
                "success": bool(command_result.get("success", False)),
                "mode": "analyze" if analyze_mode else "process",
            }
            if command_result.get("input") is not None:
                session_event["target"] = command_result.get("input")
            result_payload = command_result.get("result")
            if isinstance(result_payload, dict):
                if result_payload.get("data_type") is not None:
                    session_event["data_type"] = result_payload.get("data_type")
                if result_payload.get("output_file"):
                    session_event["output_file"] = result_payload.get("output_file")
                if result_payload.get("error"):
                    session_event["error"] = str(result_payload.get("error"))
                stats_payload = result_payload.get("statistics")
                if isinstance(stats_payload, dict) and stats_payload:
                    session_event["statistics"] = {
                        "total_rows": stats_payload.get("total_rows"),
                        "terms_standardized": stats_payload.get("terms_standardized"),
                        "values_normalized": stats_payload.get("values_normalized"),
                        "processed_files": stats_payload.get("processed_files"),
                        "failed_files": stats_payload.get("failed_files"),
                    }
            if command_result.get("stats"):
                session_event["stats"] = command_result.get("stats")
            session_summary["session_events"].append(session_event)

            if event in {"process_input", "analyze_input"}:
                session_summary["processed_inputs"] += 1
                session_summary["last_input"] = command_result.get("input")
                if command_result.get("success"):
                    session_summary["success_count"] += 1
                else:
                    session_summary["failure_count"] += 1
            elif event == "toggle_analyze":
                session_summary["analyze_requests"] += 1
            elif event == "stats":
                session_summary["stats_requests"] += 1

            if command_result.get("should_quit"):
                session_summary["quit_reason"] = "quit"
                break
        except KeyboardInterrupt:
            output_func("\n👋 再见!")
            session_summary["quit_reason"] = "keyboard_interrupt"
            session_summary["session_events"].append({
                "input": None,
                "event": "keyboard_interrupt",
                "success": True,
                "mode": "analyze" if analyze_mode else "process",
            })
            break
        except EOFError:
            output_func("\n👋 再见!")
            session_summary["quit_reason"] = "eof"
            session_summary["session_events"].append({
                "input": None,
                "event": "eof",
                "success": True,
                "mode": "analyze" if analyze_mode else "process",
            })
            break
        except Exception as e:
            output_func(f"❌ 错误: {e}")
            session_summary["failure_count"] += 1
            session_summary["last_event"] = "exception"
            session_summary["last_error"] = str(e)
            session_summary["session_events"].append({
                "input": user_input if 'user_input' in locals() else None,
                "event": "exception",
                "success": False,
                "mode": "analyze" if analyze_mode else "process",
                "error": str(e),
            })

    return session_summary


async def interactive_mode(agent: UnifiedProcessingAgent, output_dir: Optional[str] = None):
    """
    交互式处理模式

    Args:
        agent: 处理Agent
        output_dir: 输出目录
    """
    await run_interactive_session(agent=agent, output_dir=output_dir)


async def demo_csv_processing():
    """演示CSV文件处理"""
    print("\n🎯 CSV处理演示")
    print("-" * 60)
    
    # 示例CSV文件
    csv_file = "/path/to/your/data.csv"
    
    if not os.path.exists(csv_file):
        print(f"示例文件不存在: {csv_file}")
        return
    
    # 1. 分析CSV结构
    print("\n1. 分析CSV文件结构...")
    analysis = await detect_and_analyze(csv_file, verbose=True)
    
    # 2. 处理CSV文件（仅处理前10行作为演示）
    print("\n2. 处理CSV文件（演示模式，仅处理前10行）...")
    
    from processors.csv_processor import CSVProcessor
    
    processor = CSVProcessor(use_llm=True, verbose=True)
    result = await processor.process_file(
        file_path=csv_file,
        standardization_columns=["test_name", "org_name", "ab_name"],
        value_columns=["dilution_value"],
        unit_columns={"dilution_value": "dilution_text"},
        category_mapping={
            "test_name": "Test",
            "org_name": "Test",
            "ab_name": "Drug"
        },
        max_rows=10  # 演示只处理10行
    )
    
    print(f"\n处理结果:")
    print(f"  - 成功: {result.get('success')}")
    print(f"  - 处理行数: {result.get('statistics', {}).get('total_rows', 0)}")
    print(f"  - 标准化成功: {result.get('statistics', {}).get('terms_standardized', 0)}")
    
    # 显示部分结果
    if result.get("data") and len(result["data"]) > 0:
        print(f"\n示例处理结果（第一行）:")
        first_row = result["data"][0]
        for key, value in list(first_row.items())[:10]:
            print(f"  - {key}: {value}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Medical Data Cleaner - 医学数据清洗Agent框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件
  python main.py --input /path/to/file.csv
  
  # 分析文件结构（不处理）
  python main.py --input /path/to/file.csv --analyze
  
  # 批量处理目录
  python main.py --input /path/to/directory --batch
  
  # 交互模式
  python main.py --interactive
  
  # CSV处理演示
  python main.py --demo
"""
    )
    
    parser.add_argument("--input", "-i", type=str, help="输入文件路径或目录")
    parser.add_argument("--output", "-o", type=str, help="输出目录")
    parser.add_argument("--type", "-t", type=str, choices=["image", "text", "csv", "auto"],
                       default="auto", help="数据类型（默认自动检测）")
    parser.add_argument("--analyze", "-a", action="store_true", help="仅分析不处理")
    parser.add_argument("--batch", "-b", action="store_true", help="批量处理模式")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--demo", action="store_true", help="运行CSV处理演示")
    parser.add_argument("--max-files", type=int, help="批量处理最大文件数")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--no-llm", action="store_true", help="不使用LLM（仅使用本地规则）")
    parser.add_argument("--compact", "-c", action="store_true",
                       help="简洁输出模式（仅保留标准化前后对比）")
    parser.add_argument(
        "--liver-notes",
        type=str,
        nargs="?",
        const=os.path.join(
            parent_dir,
            "table", "liver_patients_note", "liver_patients_note.jsonl"
        ),
        metavar="JSONL_FILE",
        help=(
            "处理 liver_patients_note.jsonl 格式数据，对就诊文本 notes 进行信息抽取。"
            "可选择指定文件路径，默认为 table/liver_patients_note/liver_patients_note.jsonl"
        ),
    )
    parser.add_argument("--liver-max-records", type=int, default=None,
                        help="liver-notes 模式下最多处理的记录数")

    args = parser.parse_args()
    
    # 打印横幅
    print_banner()

    # --liver-notes 模式：处理 liver_patients_note.jsonl
    if args.liver_notes:
        # 显式从 medical_data_cleaner/ 本目录加载，避免被 parent_dir 中的同名模块遮蔽
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_process_liver_notes_local",
            os.path.join(current_dir, "process_liver_notes.py"),
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        batch_process_liver_notes = _mod.batch_process_liver_notes
        setup_liver_agents = _mod.setup_liver_agents
        med_agent, std_agent = setup_liver_agents()
        output_dir = args.output or os.path.join(
            parent_dir, "patient_results_liver_notes"
        )
        await batch_process_liver_notes(
            jsonl_path=args.liver_notes,
            med_agent=med_agent,
            std_agent=std_agent,
            output_dir=output_dir,
            max_records=args.liver_max_records,
            verbose=True,
        )
        return

    # 检查API配置
    config = get_api_config()
    if config['api_key']:
        print(f"✅ API配置已加载 (Base: {config['api_base']})")
    else:
        print("⚠️  未找到API Key，将使用本地规则进行处理")
    
    # 运行演示
    if args.demo:
        await demo_csv_processing()
        return
    
    # 交互模式
    if args.interactive:
        agent = UnifiedProcessingAgent(
            use_llm=not args.no_llm,
            verbose=args.verbose
        )
        await interactive_mode(agent, args.output)
        return
    
    # 需要输入路径
    if not args.input:
        parser.print_help()
        print("\n❌ 请提供输入路径 (--input)")
        return
    
    # 分析模式
    if args.analyze:
        await detect_and_analyze(args.input, verbose=True)
        return
    
    # 初始化Agent
    agent = UnifiedProcessingAgent(
        use_llm=not args.no_llm,
        verbose=args.verbose
    )
    
    # 批量处理
    if args.batch:
        await batch_process(
            args.input,
            agent,
            output_dir=args.output,
            max_files=args.max_files,
            verbose=args.verbose or True,
            compact=args.compact
        )
        return
    
    # 单文件处理
    await process_single_input(
        args.input,
        agent,
        output_dir=args.output,
        verbose=args.verbose or True,
        compact=args.compact
    )


if __name__ == "__main__":
    asyncio.run(main())
