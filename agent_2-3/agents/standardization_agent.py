# -*- coding: utf-8 -*-
"""
Medical Standardization Agent.
This agent takes extracted entities and enhances them with standard codes and normalized units.
"""
import sys
import os
import json
import re
from copy import deepcopy
from typing import Tuple, Optional

# Ensure import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from agentscope.agent import AgentBase
from agentscope.message import Msg
from schema import MedicalExtractionResult
from tools.tools_standardization import standardize_term, normalize_unit

# Try to import LLM-based normalization
try:
    from tools.tools_standardization_llm import normalize_unit_with_llm
    LLM_NORMALIZATION_AVAILABLE = True
except ImportError:
    LLM_NORMALIZATION_AVAILABLE = False


def extract_value_and_unit(value_str: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract numeric value and unit from a string like "186 mg/dL" or "8.5×10^3/mm3".
    
    Args:
        value_str: String containing value and possibly unit
    
    Returns:
        Tuple of (numeric_value, unit) or (None, None) if parsing fails
    """
    if not isinstance(value_str, str):
        return None, None
    
    value_str = value_str.strip()
    
    # Pattern for scientific notation like "8.5×10^3/mm3" or "4.5×10^12/L"
    sci_match = re.match(r'^(\d+\.?\d*)\s*[×xX]\s*10\^?(\d+)\s*(/?\w+.*)?$', value_str)
    if sci_match:
        base = float(sci_match.group(1))
        exp = int(sci_match.group(2))
        val = base * (10 ** exp)
        unit = sci_match.group(3)
        if unit:
            unit = unit.strip()
            # Handle unit like "/mm3" -> add multiplier info
            if unit.startswith('/'):
                unit = f"×10^{exp}{unit}"
        return val, unit
    
    # Pattern for regular value with unit like "186 mg/dL" or "13.2g/dL"
    match = re.match(r'^(\d+\.?\d*)\s*([a-zA-Z/%°μ×\^]+(?:/[a-zA-Z0-9%°μ×\^]+)?)?$', value_str)
    if match:
        val = float(match.group(1))
        unit = match.group(2).strip() if match.group(2) else None
        return val, unit
    
    return None, None

class MedicalStandardizationAgent(AgentBase):
    """
    Agent that processes MedicalExtractionResult to add standard codes and units.
    """
    
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        
    async def reply(self, msg: Msg | dict | None = None) -> Msg:
        """
        Process the input message containing structured extraction results.
        Expected input is a Msg object or dict with extraction results.
        """
        if msg is None:
            return Msg(self.name, "No input provided", role="assistant")
        
        # Handle Msg object
        if isinstance(msg, Msg):
            content = msg.content
            metadata = msg.metadata
        else:
            # Handle dict (for backward compatibility)
            content = msg.get("content") if isinstance(msg, dict) else None
            metadata = msg.get("metadata") if isinstance(msg, dict) else None
        
        # Determine where the data is
        data = None
        if isinstance(metadata, dict) and "entities" in metadata:
            data = metadata
        elif isinstance(content, dict) and "entities" in content:
            data = content
        elif isinstance(content, str):
            try:
                data = json.loads(content)
            except:
                pass
        
        if not data:
             return Msg(self.name, "Invalid input format. Expected structured medical data.", role="assistant")
             
        # Process entities
        processed_data = deepcopy(data)
        
        if "entities" in processed_data:
            for entity in processed_data["entities"]:
                # 1. Terminology Mapping
                # Use UMLS API if available (controlled by environment variable USE_UMLS)
                use_umls = os.environ.get("USE_UMLS", "true").lower() == "true"
                std_info = standardize_term(
                    entity["name"], 
                    entity["category"],
                    use_umls=use_umls,
                    fallback_to_umls=use_umls
                )
                if std_info:
                    entity["standard_code"] = std_info["code"]
                    entity["standard_system"] = std_info["system"]
                    # Save standard name (标准化名称)
                    if "name" in std_info:
                        entity["standard_name"] = std_info["name"]
                    # Preserve source information (umls, local_kb, llm, etc.)
                    if "source" in std_info:
                        entity["source"] = std_info["source"]
                    # Optionally update name or keep original, here we keep original and add code
                
                # 2. Unit Normalization
                # First, try to extract numeric value and unit from the value field
                # This handles cases where value is "186 mg/dL" instead of just "186"
                if entity.get("value") and isinstance(entity["value"], str):
                    extracted_val, extracted_unit = extract_value_and_unit(entity["value"])
                    if extracted_val is not None:
                        entity["value"] = extracted_val
                        if extracted_unit and not entity.get("unit"):
                            entity["unit"] = extracted_unit
                
                if entity.get("value") is not None and entity.get("unit"):
                    try:
                        val = float(entity["value"])
                        prefer_loinc = os.environ.get("LOINC_FIRST", "false").lower() == "true"
                        use_llm = os.environ.get("USE_LLM_UNIT", "true").lower() == "true"
                        
                        # Call normalize_unit with return_source=True
                        result = normalize_unit(
                            entity["name"], 
                            val, 
                            entity["unit"],
                            use_umls=use_umls,
                            return_source=True,
                            prefer_loinc=prefer_loinc
                        )
                        
                        # Unpack result (may be 2 or 3 elements)
                        if len(result) == 3:
                            norm_val, norm_unit, unit_source = result
                        else:
                            norm_val, norm_unit = result
                            unit_source = None
                        
                        # If local rules didn't convert, try LLM
                        if norm_unit == entity["unit"] and use_llm and LLM_NORMALIZATION_AVAILABLE:
                            import asyncio
                            try:
                                # 调用 LLM 进行量纲转换
                                llm_result = asyncio.get_event_loop().run_until_complete(
                                    normalize_unit_with_llm(
                                        entity["name"],
                                        val,
                                        entity["unit"],
                                        model=None,  # Will create model internally
                                        use_umls=use_umls,
                                        prefer_loinc=prefer_loinc
                                    )
                                )
                                if len(llm_result) == 3:
                                    norm_val, norm_unit, unit_source = llm_result
                            except Exception as e:
                                verbose = os.environ.get("UMLS_VERBOSE", "false").lower() == "true"
                                if verbose:
                                    print(f"[LLM量纲] 调用失败: {e}")
                        
                        if norm_unit != entity["unit"]:
                            entity["normalized_value"] = norm_val
                            entity["normalized_unit"] = norm_unit
                            if unit_source:
                                entity["unit_source"] = unit_source
                    except (ValueError, TypeError):
                        # Value might be a range or non-numeric
                        pass
                        
        return Msg(
            self.name, 
            content="Standardization completed.", 
            role="assistant",
            metadata=processed_data
        )
