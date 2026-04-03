# -*- coding: utf-8 -*-
"""
Enhanced Medical Standardization Agent using LLM.
This agent uses LLM for standardization when local KB doesn't have the mapping.
"""
import sys
import os
import json
import re
from copy import deepcopy
from typing import Optional, Tuple

# Ensure import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

src_path = os.path.abspath(os.path.join(parent_dir, "../../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from agentscope.agent import AgentBase
from agentscope.message import Msg
from schema import MedicalExtractionResult

# Try to import LLM-based standardization
try:
    from tools.tools_standardization_llm import (
        standardize_term_with_llm,
        normalize_unit_with_llm
    )
    LLM_STANDARDIZATION_AVAILABLE = True
except ImportError:
    LLM_STANDARDIZATION_AVAILABLE = False
    from tools.tools_standardization import standardize_term, normalize_unit

from agentscope.model import OpenAIChatModel


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


class EnhancedMedicalStandardizationAgent(AgentBase):
    """
    Enhanced agent that uses LLM for medical terminology standardization.
    
    This agent:
    1. First tries local knowledge base
    2. Falls back to LLM if local KB doesn't have the mapping
    3. Supports caching to avoid redundant LLM calls
    """
    
    def __init__(
        self,
        name: str,
        use_llm: bool = True,
        use_cache: bool = True,
        model: Optional[OpenAIChatModel] = None
    ) -> None:
        """
        Initialize the enhanced standardization agent.
        
        Args:
            name: Agent name
            use_llm: Whether to use LLM for standardization
            use_cache: Whether to cache standardization results
            model: LLM model instance (if None, will create one if use_llm=True)
        """
        super().__init__()
        self.name = name
        self.use_llm = use_llm and LLM_STANDARDIZATION_AVAILABLE
        self.use_cache = use_cache
        self.model = model
        
        # Initialize model if needed
        if self.use_llm and self.model is None:
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
            api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            if api_key:
                try:
                    self.model = OpenAIChatModel(
                        model_name="gpt-4.1-mini",  # Use cost-effective model
                        api_key=api_key,
                        client_kwargs={"base_url": api_base},
                        generate_kwargs={"temperature": 0.0}
                    )
                except Exception:
                    self.use_llm = False
        
        # Fallback to local functions if LLM not available
        if not self.use_llm:
            from tools.tools_standardization import standardize_term, normalize_unit
            self._standardize_term = standardize_term
            self._normalize_unit = normalize_unit
            self._use_async = False
        else:
            self._use_async = True  # Mark that we need async calls
        
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
        standardization_stats = {
            "total": 0,
            "standardized": 0,
            "unit_normalized": 0
        }
        
        if "entities" in processed_data:
            for entity in processed_data["entities"]:
                standardization_stats["total"] += 1
                
                # 1. Terminology Mapping
                # Priority: LLM (if enabled) > UMLS > Local KB
                # Note: standardize_term() now tries UMLS first, then falls back to Local KB
                if self._use_async:
                    std_info = await standardize_term_with_llm(
                        entity["name"], 
                        entity["category"],
                        use_cache=self.use_cache,
                        model=self.model,
                        fallback_to_local=False  # We'll handle UMLS fallback separately
                    )
                    # If LLM didn't find it, try UMLS
                    if not std_info or not std_info.get("code"):
                        from tools.tools_standardization import standardize_term
                        use_umls = os.environ.get("USE_UMLS", "true").lower() == "true"
                        std_info = standardize_term(
                            entity["name"],
                            entity["category"],
                            use_umls=use_umls,
                            fallback_to_umls=use_umls
                        )
                else:
                    use_umls = os.environ.get("USE_UMLS", "true").lower() == "true"
                    std_info = self._standardize_term(
                        entity["name"], 
                        entity["category"],
                        use_umls=use_umls,
                        fallback_to_umls=use_umls
                    )
                    
                if std_info and std_info.get("code"):
                    entity["standard_code"] = std_info["code"]
                    entity["standard_system"] = std_info["system"]
                    # Preserve standard name (标准术语名称)
                    if "name" in std_info:
                        entity["standard_name"] = std_info["name"]
                    # Preserve CUI (UMLS Concept Unique Identifier)
                    if "cui" in std_info:
                        entity["cui"] = std_info["cui"]
                    # Preserve source information (umls, local_kb, llm, etc.)
                    if "source" in std_info:
                        entity["source"] = std_info["source"]
                    standardization_stats["standardized"] += 1
                
                # Unit normalization disabled
                        
        # Add statistics to metadata
        processed_data["_standardization_stats"] = standardization_stats
        
        return Msg(
            self.name, 
            content=f"Standardization completed. {standardization_stats['standardized']}/{standardization_stats['total']} entities standardized, {standardization_stats['unit_normalized']} units normalized.", 
            role="assistant",
            metadata=processed_data
        )
