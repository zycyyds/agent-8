# -*- coding: utf-8 -*-
"""
Medical Terminology Standardization Tools using LLM.
This module provides functions to standardize medical terms using LLM APIs
when local knowledge base doesn't have the mapping.
"""
import os
import json
import sys
from typing import Dict, Optional, Tuple, Any
from functools import lru_cache


try:
    from agentscope.model import OpenAIChatModel
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Local cache for standardization results
_STANDARDIZATION_CACHE: Dict[str, Dict[str, Any]] = {}

# Mapping from category to standard coding system
CATEGORY_TO_SYSTEM = {
    "Disease": "ICD-10",
    "Drug": "ATC",
    "Test": "LOINC",
    "Symptom": "SNOMED-CT",
    "Treatment": "SNOMED-CT",
    "Anatomy": "SNOMED-CT",
    "LabValue": "LOINC"
}

async def _query_llm_for_standardization(
    term: str,
    category: str,
    model: Optional[Any] = None
) -> Dict[str, str]:
    """
    Query LLM to get standardization information for a medical term.
    
    Args:
        term: The medical term to standardize
        category: The category of the term (Disease, Drug, Test, etc.)
        model: The LLM model instance (if None, will try to create one)
    
    Returns:
        Dict with 'code', 'system', 'name' if found, empty dict otherwise
    """
    if not LLM_AVAILABLE:
        return {}
    
    # Try to get model from parameter or environment
    if model is None:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        if not api_key:
            return {}
        
        try:
            model = OpenAIChatModel(
                model_name="gpt-4.1-mini",  # Use smaller model for cost efficiency
                api_key=api_key,
                client_kwargs={"base_url": api_base},
                generate_kwargs={"temperature": 0.0}
            )
        except Exception:
            return {}
    
    coding_system = CATEGORY_TO_SYSTEM.get(category, "ICD-10")
    
    # Construct prompt for LLM
    prompt = f"""你是一个医疗术语标准化专家。请将以下医疗术语标准化为{coding_system}编码系统。

术语: {term}
类别: {category}
编码系统: {coding_system}

请返回JSON格式，包含以下字段：
- code: 标准编码（如果没有找到，返回null）
- system: 编码系统名称（{coding_system}）
- name: 标准的英文或中文名称

如果无法找到对应的标准编码，请返回 {{"code": null, "system": "{coding_system}", "name": null}}

只返回JSON，不要其他文字。"""

    try:
        response = await model([{"role": "user", "content": prompt}])
        
        # Parse response
        if hasattr(response, 'content') and response.content:
            content = response.content[0].get("text", "") if isinstance(response.content, list) else str(response.content)
            
            # Try to extract JSON from response
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            try:
                result = json.loads(content)
                if result.get("code") and result.get("code") != "null":
                    return {
                        "code": str(result["code"]),
                        "system": result.get("system", coding_system),
                        "name": result.get("name", term)
                    }
            except json.JSONDecodeError:
                pass
    except Exception as e:
        # Silently fail and return empty dict
        pass
    
    return {}


async def standardize_term_with_llm(
    term: str,
    category: str,
    use_cache: bool = True,
    model: Optional[Any] = None,
    fallback_to_local: bool = True
) -> Dict[str, str]:
    """
    Standardize a medical term using LLM with optional local cache.
    
    This function uses a hybrid approach:
    1. Check local cache first (if use_cache=True)
    2. Query LLM if not in cache
    3. Fall back to local KB if LLM fails and fallback_to_local=True
    
    Args:
        term: The medical term to standardize
        category: The category (Disease, Drug, Test, etc.)
        use_cache: Whether to use cached results
        model: LLM model instance (optional)
        fallback_to_local: Whether to fall back to local KB
    
    Returns:
        Dict with 'code', 'system', 'name' if found, empty dict otherwise
    """
    if not term or not category:
        return {}
    
    # Check cache
    cache_key = f"{category}::{term}"
    if use_cache and cache_key in _STANDARDIZATION_CACHE:
        return _STANDARDIZATION_CACHE[cache_key]
    
    # Try LLM query
    result = await _query_llm_for_standardization(term, category, model)
    
    # Fall back to local KB or UMLS if LLM failed
    if not result and fallback_to_local:
        from tools.tools_standardization import standardize_term as local_standardize
        # Use UMLS if available (controlled by environment variable)
        use_umls = os.environ.get("USE_UMLS", "true").lower() == "true"
        result = local_standardize(term, category, use_umls=use_umls, fallback_to_umls=use_umls)
    
    # Cache result
    if result and use_cache:
        _STANDARDIZATION_CACHE[cache_key] = result
    
    return result


# Unit conversion rules cache (learned from LLM)
_UNIT_CONVERSION_CACHE: Dict[str, Dict[str, Any]] = {}


def _safe_eval_formula(formula: str, value: float) -> Optional[float]:
    """
    安全地执行数学公式计算。
    
    Args:
        formula: 转换公式，如 "x * 18.016" 或 "(x - 32) * 5 / 9"
        value: 输入值
    
    Returns:
        计算结果，如果公式无效则返回 None
    """
    import math
    
    # 替换 x 为实际值
    formula = formula.replace('x', str(value))
    formula = formula.replace('X', str(value))
    
    # 定义允许的函数和变量
    allowed_names = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'pow': pow,
        'sqrt': math.sqrt,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e
    }
    
    try:
        # 安全执行公式
        result = eval(formula, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception as e:
        print(f"[量纲转换] 公式执行失败: {formula}, 错误: {e}")
        return None


async def _query_llm_for_unit_conversion(
    test_name: str,
    from_unit: str,
    to_unit: Optional[str],
    model: Any,
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    调用大模型获取单位转换规则。
    
    Args:
        test_name: 检验项目名称
        from_unit: 原单位
        to_unit: 目标单位（可选，如果为 None 则让 LLM 推荐标准单位）
        model: LLM 模型实例
        verbose: 是否打印详细信息
    
    Returns:
        转换规则字典，包含 formula, to_unit, description 等
    """
    if to_unit:
        prompt = f"""你是一个医疗检验单位转换专家。请提供以下单位转换的公式。

检验项目: {test_name}
原单位: {from_unit}
目标单位: {to_unit}

请返回JSON格式：
{{
    "formula": "<转换公式，用x表示原值>",
    "to_unit": "{to_unit}",
    "description": "<转换说明>",
    "example": "<例如：100 {from_unit} = ? {to_unit}>"
}}

例如，如果是 mg/dL 转 mmol/L (血糖)，公式是 "x / 18.016"

只返回JSON，不要其他文字。如果无法确定转换公式，请在formula字段返回null。"""
    else:
        prompt = f"""你是一个医疗检验单位转换专家。请为以下检验项目推荐标准单位并提供转换公式。

检验项目: {test_name}
原单位: {from_unit}

请返回JSON格式：
{{
    "formula": "<转换公式，用x表示原值>",
    "to_unit": "<推荐的标准单位（SI单位或医学常用标准单位）>",
    "description": "<转换说明>",
    "example": "<例如：100 {from_unit} = ? <目标单位>>"
}}

医学常用标准单位参考：
- 血糖: mmol/L
- 血红蛋白: g/L
- 白细胞: 10^9/L
- 血小板: 10^9/L
- 胆固醇: mmol/L
- 肌酐: μmol/L
- 尿素: mmol/L
- 平均红细胞血红蛋白含量(MCH): pg (皮克，不需要转换)
- 平均红细胞体积(MCV): fL (飞升，不需要转换)

【重要】以下单位已经是医学临床标准，不需要转换，请返回formula为null：
- 角度/角速度: °, 度, 度/秒, °/s (眼科检查标准单位)
- 血液学: pg, fL, % (血常规指标标准单位)
- 已是SI单位: mmol/L, μmol/L, g/L, ×10^9/L, ×10^12/L

只返回JSON，不要其他文字。如果原单位已经是标准单位或无法确定转换公式，请在formula字段返回null。"""

    try:
        if verbose:
            print(f"[LLM量纲] 查询: {test_name} ({from_unit} -> {to_unit or '标准单位'})")
        
        # 直接使用 OpenAI 客户端调用
        import openai
        
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        # 提取响应内容
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
        else:
            content = ""
        
        content = content.strip()
        
        # 提取 JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        content = content.strip()
        
        try:
            result = json.loads(content)
            if result.get("formula") and result["formula"] != "null":
                if verbose:
                    print(f"[LLM量纲] 获取到转换规则: {result}")
                return result
            elif verbose:
                print(f"[LLM量纲] LLM返回null或无法确定转换")
        except json.JSONDecodeError as e:
            if verbose:
                print(f"[LLM量纲] JSON解析失败: {e}, 内容: {content[:200]}")
    except Exception as e:
        if verbose:
            print(f"[LLM量纲] 查询失败: {e}")
    
    return None


async def normalize_unit_with_llm(
    test_name: str,
    value: float,
    unit: str,
    model: Optional[Any] = None,
    use_umls: bool = True,
    loinc_code: Optional[str] = None,
    prefer_loinc: bool = None,
    verbose: bool = None
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    使用 LLM 进行智能量纲统一。
    
    当本地规则和 UMLS API 都无法找到转换规则时，调用大模型获取转换公式，
    然后使用 Python 进行计算。
    
    Strategy (Priority Order):
    1. 本地规则 / UMLS API（根据 prefer_loinc 设置）
    2. 检查 LLM 转换规则缓存
    3. 调用 LLM 获取新的转换规则
    4. 使用 Python 执行转换公式
    
    Args:
        test_name: 检验项目名称
        value: 数值
        unit: 原单位
        model: LLM 模型实例（可选，如果为 None 会尝试创建）
        use_umls: 是否使用 UMLS API
        loinc_code: LOINC 代码（可选）
        prefer_loinc: 是否优先使用 LOINC
        verbose: 是否打印详细信息（默认从环境变量读取）
    
    Returns:
        Tuple of (normalized_value, normalized_unit, source)
        source: "local", "umls", "llm_cached", "llm", 或 None
    """
    if verbose is None:
        verbose = os.environ.get("UMLS_VERBOSE", "false").lower() == "true"
    
    # Check if we should prefer LOINC (from parameter or environment variable)
    if prefer_loinc is None:
        prefer_loinc = os.environ.get("LOINC_FIRST", "false").lower() == "true"
    
    # Step 1: 尝试本地规则 / UMLS API
    from tools.tools_standardization import normalize_unit as local_normalize
    result = local_normalize(
        test_name, value, unit, 
        use_umls=use_umls, 
        loinc_code=loinc_code,
        return_source=True,
        prefer_loinc=prefer_loinc
    )
    
    # Unpack result
    if len(result) == 3:
        norm_val, norm_unit, source = result
    else:
        norm_val, norm_unit = result
        source = None
    
    # If already normalized, return
    if norm_unit != unit and source:
        return norm_val, norm_unit, source
    
    # Step 2: 检查 LLM 转换规则缓存
    cache_key = f"{test_name}::{unit}"
    if cache_key in _UNIT_CONVERSION_CACHE:
        cached_rule = _UNIT_CONVERSION_CACHE[cache_key]
        formula = cached_rule.get("formula")
        to_unit = cached_rule.get("to_unit")
        
        if formula and to_unit:
            converted_value = _safe_eval_formula(formula, value)
            if converted_value is not None:
                if verbose:
                    print(f"[LLM量纲] 使用缓存规则: {value} {unit} -> {converted_value} {to_unit}")
                return converted_value, to_unit, "llm_cached"
    
    # Step 3: 如果没有模型，尝试创建一个
    if model is None and LLM_AVAILABLE:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        if api_key:
            try:
                model = OpenAIChatModel(
                    model_name="gpt-4.1-mini",  # 使用小模型以降低成本
                    api_key=api_key,
                    client_kwargs={"base_url": api_base},
                    generate_kwargs={"temperature": 0.0, "stream": False}
                )
            except Exception as e:
                if verbose:
                    print(f"[LLM量纲] 创建模型失败: {e}")
    
    # Step 4: 调用 LLM 获取转换规则
    if model:
        conversion_rule = await _query_llm_for_unit_conversion(
            test_name, unit, None, model, verbose
        )
        
        if conversion_rule:
            formula = conversion_rule.get("formula")
            to_unit = conversion_rule.get("to_unit")
            
            if formula and to_unit:
                # 缓存规则
                _UNIT_CONVERSION_CACHE[cache_key] = conversion_rule
                
                # 执行转换
                converted_value = _safe_eval_formula(formula, value)
                if converted_value is not None:
                    # 四舍五入到合适精度
                    if abs(converted_value) >= 1:
                        converted_value = round(converted_value, 2)
                    else:
                        converted_value = round(converted_value, 4)
                    
                    if verbose:
                        print(f"[LLM量纲] 转换成功: {value} {unit} -> {converted_value} {to_unit}")
                        print(f"[LLM量纲] 公式: {formula}")
                    
                    return converted_value, to_unit, "llm"
    
    # Return original if all methods fail
    if verbose:
        print(f"[LLM量纲] 无法转换: {test_name} ({value} {unit})")
    return value, unit, None


def get_llm_conversion_cache() -> Dict[str, Dict[str, Any]]:
    """获取 LLM 转换规则缓存。"""
    return _UNIT_CONVERSION_CACHE.copy()


def clear_llm_conversion_cache() -> None:
    """清除 LLM 转换规则缓存。"""
    global _UNIT_CONVERSION_CACHE
    _UNIT_CONVERSION_CACHE.clear()


def clear_standardization_cache() -> None:
    """Clear the standardization cache."""
    global _STANDARDIZATION_CACHE
    _STANDARDIZATION_CACHE.clear()


def get_cache_stats() -> Dict[str, int]:
    """Get statistics about the cache."""
    return {
        "total_entries": len(_STANDARDIZATION_CACHE),
        "by_category": {}
    }
