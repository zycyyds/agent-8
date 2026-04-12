# -*- coding: utf-8 -*-
"""
文件夹分析工具 - 递归遍历和病人ID识别

支持的功能：
1. 递归扫描文件夹结构
2. 自动识别病人ID（从文件夹名或文件名）
3. 按病人分组文件
4. 分析文件夹层级结构
"""

import os
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict


# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

# 病人ID模式（数字ID）
PATIENT_ID_PATTERN = re.compile(r'^(\d{4,8})$')  # 4-8位数字


def _get_skip_folders() -> Set[str]:
    """获取需要跳过的文件夹名称"""
    try:
        from config.settings import get_skip_folders
        return get_skip_folders()
    except ImportError:
        # fallback: 直接从环境变量读取
        env_value = os.environ.get("SKIP_FOLDERS")
        if env_value is not None:
            if env_value.strip() == "":
                return set()
            return {name.strip() for name in env_value.split(",") if name.strip()}
        return {'figure', '分割'}


def scan_folder_structure(
    root_path: str,
    max_depth: Optional[int] = None,
    include_hidden: bool = False
) -> Dict[str, Any]:
    """
    扫描文件夹结构
    
    Args:
        root_path: 根目录路径
        max_depth: 最大扫描深度（None表示无限制）
        include_hidden: 是否包含隐藏文件/文件夹
        
    Returns:
        文件夹结构信息
    """
    if not os.path.exists(root_path):
        return {"error": f"路径不存在: {root_path}"}
    
    if os.path.isfile(root_path):
        return {
            "type": "file",
            "path": root_path,
            "name": os.path.basename(root_path),
            "is_image": _is_image_file(root_path)
        }
    
    structure = {
        "type": "directory",
        "path": root_path,
        "name": os.path.basename(root_path) or root_path,
        "children": [],
        "statistics": {
            "total_files": 0,
            "image_files": 0,
            "subdirectories": 0,
            "max_depth": 0
        }
    }
    
    _scan_recursive(structure, root_path, 0, max_depth, include_hidden)
    
    return structure


def _scan_recursive(
    parent: Dict,
    current_path: str,
    current_depth: int,
    max_depth: Optional[int],
    include_hidden: bool
) -> None:
    """递归扫描目录"""
    if max_depth is not None and current_depth >= max_depth:
        return
    
    skip_folders = _get_skip_folders()
    
    try:
        entries = sorted(os.listdir(current_path))
    except PermissionError:
        return
    
    for entry in entries:
        # 跳过隐藏文件
        if not include_hidden and entry.startswith('.'):
            continue
        
        entry_path = os.path.join(current_path, entry)
        
        if os.path.isfile(entry_path):
            is_image = _is_image_file(entry_path)
            child = {
                "type": "file",
                "name": entry,
                "path": entry_path,
                "is_image": is_image
            }
            parent["children"].append(child)
            parent["statistics"]["total_files"] += 1
            if is_image:
                parent["statistics"]["image_files"] += 1
                
        elif os.path.isdir(entry_path):
            # 跳过配置中指定的文件夹
            if entry in skip_folders:
                continue
            
            child = {
                "type": "directory",
                "name": entry,
                "path": entry_path,
                "children": [],
                "statistics": {
                    "total_files": 0,
                    "image_files": 0,
                    "subdirectories": 0,
                    "max_depth": 0
                }
            }
            
            _scan_recursive(child, entry_path, current_depth + 1, max_depth, include_hidden)
            
            parent["children"].append(child)
            parent["statistics"]["subdirectories"] += 1
            parent["statistics"]["total_files"] += child["statistics"]["total_files"]
            parent["statistics"]["image_files"] += child["statistics"]["image_files"]
            parent["statistics"]["max_depth"] = max(
                parent["statistics"]["max_depth"],
                child["statistics"]["max_depth"] + 1
            )


def _is_image_file(file_path: str) -> bool:
    """判断是否是图片文件"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in IMAGE_EXTENSIONS


def extract_patient_id(name: str) -> Optional[str]:
    """
    从文件名或文件夹名提取病人ID
    
    Args:
        name: 文件名或文件夹名
        
    Returns:
        病人ID或None
    """
    # 直接匹配纯数字文件夹名
    match = PATIENT_ID_PATTERN.match(name)
    if match:
        return match.group(1)
    
    # 从文件名中提取（如 36907(1).jpg -> 36907）
    # 匹配开头的数字
    match = re.match(r'^(\d{4,8})', name)
    if match:
        return match.group(1)
    
    return None


def group_files_by_patient(
    root_path: str,
    include_hidden: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    按病人ID分组文件
    
    Args:
        root_path: 根目录路径
        include_hidden: 是否包含隐藏文件
        
    Returns:
        按病人ID分组的文件信息
        {
            "36907": {
                "patient_id": "36907",
                "files": [
                    {"path": "...", "category": "实验室检查", "subcategory": "..."},
                    ...
                ],
                "categories": ["实验室检查", "垂直眼位", ...]
            },
            ...
        }
    """
    if not os.path.isdir(root_path):
        if os.path.isfile(root_path):
            # 单个文件
            patient_id = extract_patient_id(os.path.basename(root_path))
            if patient_id:
                return {
                    patient_id: {
                        "patient_id": patient_id,
                        "files": [{"path": root_path, "category": "unknown"}],
                        "categories": ["unknown"]
                    }
                }
            else:
                return {
                    "unknown": {
                        "patient_id": "unknown",
                        "files": [{"path": root_path, "category": "unknown"}],
                        "categories": ["unknown"]
                    }
                }
        return {}
    
    patient_files = defaultdict(lambda: {
        "patient_id": None,
        "files": [],
        "categories": set()
    })
    
    # 跳过的目录名（这些不是分类名）
    skip_dirs = {'ocr', 'figure', 'data', 'images', 'extracted', 'raw', '归档'}
    
    # 预先分析 root_path 本身，提取可能的 category
    # 例如: .../归档/data/ocr/垂直眼位/36906 -> category = "垂直眼位"
    root_parts = os.path.normpath(root_path).split(os.sep)
    root_category = None
    root_patient_id = None
    
    # 从后往前分析 root_path
    for i in range(len(root_parts) - 1, -1, -1):
        part = root_parts[i]
        if not part or part in ['.', '..']:
            continue
        
        pid = extract_patient_id(part)
        if pid:
            root_patient_id = pid
            # 病人ID的上一级目录就是 category
            if i > 0 and root_parts[i-1].lower() not in skip_dirs:
                root_category = root_parts[i-1]
            break
        elif part.lower() not in skip_dirs:
            # 可能是 category
            root_category = part
    
    # 获取需要跳过的文件夹
    skip_folders = _get_skip_folders()
    
    # 递归遍历
    for dirpath, dirnames, filenames in os.walk(root_path):
        # 过滤隐藏目录和配置中指定跳过的文件夹
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in skip_folders]
        else:
            dirnames[:] = [d for d in dirnames if d not in skip_folders]
        
        # 分析路径层级（相对于 root_path）
        rel_path = os.path.relpath(dirpath, root_path)
        path_parts = rel_path.split(os.sep) if rel_path != '.' else []
        
        # 从相对路径识别 category 和 patient_id
        category = root_category  # 默认使用从 root_path 提取的
        subcategory = None
        patient_id_from_path = root_patient_id  # 默认使用从 root_path 提取的
        
        for i, part in enumerate(path_parts):
            if part in ['.', '..']:
                continue
                
            pid = extract_patient_id(part)
            if pid:
                # 找到病人ID，其上一级是 category
                patient_id_from_path = pid
                if i > 0 and path_parts[i-1].lower() not in skip_dirs:
                    category = path_parts[i-1]
            elif part.lower() in skip_dirs:
                continue
            else:
                # 这可能是分类名（如"实验室检查"、"垂直眼位"）
                if category is None or category == root_category:
                    category = part
        
        # 处理文件
        for filename in filenames:
            if not include_hidden and filename.startswith('.'):
                continue
            
            file_path = os.path.join(dirpath, filename)
            
            if not _is_image_file(file_path):
                continue
            
            # 尝试从文件名提取病人ID
            patient_id = extract_patient_id(filename)
            if not patient_id:
                patient_id = patient_id_from_path
            if not patient_id:
                patient_id = "unknown"
            
            # 检测图像来源类型（ocr 或 figure）
            source_type = "ocr"  # 默认为 ocr
            path_parts_full = os.path.normpath(file_path).split(os.sep)
            for part in path_parts_full:
                if part.lower() == 'figure':
                    source_type = "figure"
                    break
            
            file_info = {
                "path": file_path,
                "filename": filename,
                "category": category or "unknown",
                "subcategory": subcategory,
                "relative_path": os.path.relpath(file_path, root_path),
                "source_type": source_type  # 'ocr' 或 'figure'
            }
            
            patient_files[patient_id]["patient_id"] = patient_id
            patient_files[patient_id]["files"].append(file_info)
            if category:
                patient_files[patient_id]["categories"].add(category)
    
    # 转换set为list
    for pid in patient_files:
        patient_files[pid]["categories"] = sorted(list(patient_files[pid]["categories"]))
        patient_files[pid]["file_count"] = len(patient_files[pid]["files"])
    
    return dict(patient_files)


def analyze_folder_for_agent(root_path: str) -> Dict[str, Any]:
    """
    为Agent提供的文件夹分析接口
    
    返回结构化的分析结果，包括：
    - 文件夹结构概览
    - 识别到的病人列表
    - 每个病人的文件分组
    - 建议的处理方式
    
    Args:
        root_path: 要分析的路径
        
    Returns:
        分析结果
    """
    result = {
        "root_path": root_path,
        "is_directory": os.path.isdir(root_path),
        "is_file": os.path.isfile(root_path),
        "exists": os.path.exists(root_path)
    }
    
    if not result["exists"]:
        result["error"] = f"路径不存在: {root_path}"
        return result
    
    if result["is_file"]:
        # 单个文件
        result["type"] = "single_file"
        result["is_image"] = _is_image_file(root_path)
        patient_id = extract_patient_id(os.path.basename(root_path))
        result["patient_id"] = patient_id
        result["recommendation"] = "process_single_file"
        return result
    
    # 目录分析
    result["type"] = "directory"
    
    # 扫描结构
    structure = scan_folder_structure(root_path, max_depth=5)
    result["statistics"] = structure.get("statistics", {})
    
    # 按病人分组
    patient_groups = group_files_by_patient(root_path)
    result["patient_count"] = len(patient_groups)
    result["patient_ids"] = sorted(patient_groups.keys())
    result["patients"] = patient_groups
    
    # 生成建议
    if result["patient_count"] == 0:
        result["recommendation"] = "no_processable_files"
    elif result["patient_count"] == 1 and "unknown" in patient_groups:
        result["recommendation"] = "process_as_single_batch"
    else:
        result["recommendation"] = "process_by_patient"
    
    # 生成处理计划
    result["processing_plan"] = []
    for pid, pdata in sorted(patient_groups.items()):
        plan_item = {
            "patient_id": pid,
            "file_count": pdata["file_count"],
            "categories": pdata["categories"],
            "files": [f["path"] for f in pdata["files"]]
        }
        result["processing_plan"].append(plan_item)
    
    return result


def get_folder_summary(root_path: str) -> str:
    """
    获取文件夹的文本摘要，适合作为Agent的输入
    
    Args:
        root_path: 文件夹路径
        
    Returns:
        文本摘要
    """
    analysis = analyze_folder_for_agent(root_path)
    
    if "error" in analysis:
        return f"错误: {analysis['error']}"
    
    if analysis["type"] == "single_file":
        return f"单个文件: {root_path}\n是否图片: {analysis['is_image']}\n病人ID: {analysis.get('patient_id', '未知')}"
    
    lines = [
        f"目录分析: {root_path}",
        f"",
        f"统计信息:",
        f"  - 总文件数: {analysis['statistics'].get('total_files', 0)}",
        f"  - 图片文件: {analysis['statistics'].get('image_files', 0)}",
        f"  - 子目录数: {analysis['statistics'].get('subdirectories', 0)}",
        f"  - 目录深度: {analysis['statistics'].get('max_depth', 0)}",
        f"",
        f"识别到 {analysis['patient_count']} 个病人:",
    ]
    
    for pid in analysis["patient_ids"]:
        pdata = analysis["patients"][pid]
        categories = ", ".join(pdata["categories"]) if pdata["categories"] else "无分类"
        lines.append(f"  - 病人 {pid}: {pdata['file_count']} 个文件 ({categories})")
    
    lines.extend([
        f"",
        f"建议处理方式: {analysis['recommendation']}"
    ])
    
    return "\n".join(lines)


# Agent工具函数
def analyze_input_path(path: str) -> Dict[str, Any]:
    """
    分析输入路径，判断是文件还是目录，并返回处理建议
    
    这是一个Agent工具函数。
    
    Args:
        path: 输入路径（可以是文件或目录）
        
    Returns:
        分析结果，包含路径类型、病人分组、处理建议
    """
    return analyze_folder_for_agent(path)


def list_patient_files(path: str, patient_id: str) -> List[Dict[str, Any]]:
    """
    获取指定病人的所有文件
    
    这是一个Agent工具函数。
    
    Args:
        path: 根目录路径
        patient_id: 病人ID
        
    Returns:
        该病人的所有文件列表
    """
    patient_groups = group_files_by_patient(path)
    
    if patient_id in patient_groups:
        return patient_groups[patient_id]["files"]
    
    return []


if __name__ == "__main__":
    # 测试
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = "/path/to/your/data"
    
    print(get_folder_summary(test_path))
    print("\n" + "=" * 60 + "\n")
    
    analysis = analyze_folder_for_agent(test_path)
    import json
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
