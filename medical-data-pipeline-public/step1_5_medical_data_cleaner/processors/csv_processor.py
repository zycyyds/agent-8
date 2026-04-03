# -*- coding: utf-8 -*-
"""
CSV数据处理器

处理CSV结构化数据，支持以下处理模式：
1. 直接标准化和量纲统一（针对已结构化的列）
2. 预处理 + 信息抽取 + 标准化（针对包含自由文本的列，如放射报告、出院记录等）
"""
import os
import sys
import csv
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# 设置导入路径
module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

# 获取上级目录（medical_agent_system）以导入标准化工具
parent_dir = os.path.dirname(module_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

src_path = os.path.abspath(os.path.join(parent_dir, "../../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tools.csv_reader import read_csv_data, analyze_columns_for_standardization, read_csv_sample


class CSVProcessor:
    """
    CSV数据处理器
    
    功能：
    1. 读取CSV文件并分析结构
    2. 自动或手动指定需要标准化/量纲统一的列
    3. 对指定列进行标准化和量纲统一
    4. 对指定的文本列进行信息抽取（支持预处理 -> 信息抽取 -> 标准化流程）
    5. 输出处理后的数据
    
    处理模式：
    - standardization_columns: 已结构化的术语列，直接进行标准化
    - value_columns + unit_columns: 数值列进行量纲统一
    - extraction_columns: 自由文本列，需要先进行信息抽取再标准化
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        use_umls: bool = True,
        model: Optional[Any] = None,
        verbose: bool = False
    ):
        """
        初始化CSV处理器
        
        Args:
            use_llm: 是否使用LLM进行标准化和信息抽取
            use_umls: 是否使用UMLS API
            model: LLM模型实例
            verbose: 是否打印详细信息
        """
        self.use_llm = use_llm
        self.use_umls = use_umls
        self.model = model
        self.verbose = verbose
        
        # 信息抽取相关
        self.med_agent = None
        self.std_agent = None
        self.extraction_schema = None
        
        # 导入标准化工具
        self._import_standardization_tools()
        
        # 初始化信息抽取Agent（如果使用LLM）
        if use_llm:
            self._init_extraction_agents()
    
    def _import_standardization_tools(self):
        """导入标准化工具"""
        try:
            # 尝试导入LLM标准化工具
            try:
                from tools.standardization import (
                    standardize_term_with_llm,
                    normalize_unit_with_llm
                )
            except ImportError:
                from tools.tools_standardization_llm import (
                    standardize_term_with_llm,
                    normalize_unit_with_llm
                )
            self.standardize_term = standardize_term_with_llm
            self.normalize_unit = normalize_unit_with_llm
            self._async_mode = True
            if self.verbose:
                print("[CSV处理器] 使用LLM标准化工具")
        except ImportError:
            # 回退到本地标准化工具
            try:
                from tools.standardization import standardize_term, normalize_unit
            except ImportError:
                from tools.tools_standardization import standardize_term, normalize_unit
            self.standardize_term = standardize_term
            self.normalize_unit = normalize_unit
            self._async_mode = False
            if self.verbose:
                print("[CSV处理器] 使用本地标准化工具")
    
    def _init_extraction_agents(self):
        """初始化信息抽取Agent"""
        try:
            from config.settings import get_api_config
            from agentscope.model import OpenAIChatModel
            from agentscope.agent import ReActAgent
            from agentscope.formatter import OpenAIChatFormatter
            from agentscope.tool import Toolkit
            
            config = get_api_config()
            if not config.get('api_key'):
                if self.verbose:
                    print("[CSV处理器] 未找到API Key，跳过信息抽取Agent初始化")
                return
            
            # 初始化模型（如果尚未提供）
            if not self.model:
                self.model = OpenAIChatModel(
                    config_name="csv-processor-model",
                    model_name=config['model_name'],
                    api_key=config['api_key'],
                    client_kwargs={
                        "base_url": config['api_base'],
                        "timeout": config['timeout'],
                    },
                    generate_kwargs={"temperature": 0.0}
                )
            
            # 创建工具包
            try:
                from tools.tools_preprocess import preprocess_medical_input
                from agentscope.tool._text_processing._medical_clean import clean_medical_text
                
                toolkit = Toolkit()
                toolkit.create_tool_group(
                    group_name="medical",
                    description="Medical data processing tools",
                    active=True,
                )
                toolkit.register_tool_function(
                    preprocess_medical_input,
                    group_name="medical",
                    func_description="Preprocess medical text input."
                )
                toolkit.register_tool_function(
                    clean_medical_text,
                    group_name="medical",
                    func_description="Clean medical text by normalizing symbols and whitespace."
                )
            except ImportError as e:
                if self.verbose:
                    print(f"[CSV处理器] 工具导入失败: {e}")
                toolkit = None
            
            # 创建医学信息提取Agent
            extraction_prompt = """You are a professional medical data extraction expert. Extract structured information from medical text in **concise key-value pairs**.

## CRITICAL: Language Rule
- **MUST keep the SAME language as the input text**
- If input is English, output in English
- If input is Chinese, output in Chinese
- NEVER translate the content

## Core Principles
- **Concise**: Keep name and value short
- **Key-value pairs**: Format as "location/item: finding/result"
- **Split**: Break complex descriptions into multiple short entities

## Entity Categories

### Test (Examination)
- name: Test name (short), e.g., "Chest X-ray", "Liver ultrasound"
- value: Leave empty or "completed"

### Finding (Examination findings) ⭐Most important
- name: Anatomical location or object (short), e.g., "Liver", "Gallbladder", "Left lung", "Heart"
- value: Finding result (short), e.g., "increased echo", "small effusion", "normal", "nodule"

### Disease (Diagnosis)
- name: Disease name (short), e.g., "Pneumonia", "Gallstones"
- value: Stage/severity if any, e.g., "mild", "Stage III"

### Anatomy (Anatomical location)
- name: Location name, e.g., "Left thoracic cavity"
- value: Status description (short)

## Examples

### English Input:
"No focal consolidation. No pleural effusion or pneumothorax. The cardiomediastinal silhouette is normal."

Correct Output:
{
    "entities": [
        {"category": "Finding", "name": "Lung", "value": "no focal consolidation"},
        {"category": "Finding", "name": "Pleural space", "value": "no effusion"},
        {"category": "Finding", "name": "Lung", "value": "no pneumothorax"},
        {"category": "Finding", "name": "Cardiomediastinal silhouette", "value": "normal"}
    ],
    "impression": "No acute cardiopulmonary abnormality"
}

### Chinese Input:
"肝脏回声增粗，呈结节样改变。未见局灶性肝占位性病变。少量腹水。"

Correct Output:
{
    "entities": [
        {"category": "Finding", "name": "肝脏", "value": "回声增粗"},
        {"category": "Finding", "name": "肝脏", "value": "结节样改变"},
        {"category": "Finding", "name": "肝脏占位", "value": "未见"},
        {"category": "Finding", "name": "腹腔", "value": "少量腹水"}
    ]
}

## Output Format
{
    "entities": [
        {"category": "Category", "name": "short_name", "value": "short_result"}
    ],
    "impression": "Overall impression (one sentence)",
    "indication": "Indication if any"
}

Return JSON only."""

            self.med_agent = ReActAgent(
                name="CSVMedicalExpert",
                sys_prompt=extraction_prompt,
                model=self.model,
                formatter=OpenAIChatFormatter(),
                toolkit=toolkit if toolkit else None,
            )
            
            # 创建标准化Agent
            standardization_prompt = """你是医学术语标准化专家。对提取的医学实体进行标准化：

1. 术语标准化：将医学术语映射到标准编码（如SNOMED-CT、LOINC、ICD-10等）
2. 量纲统一：将数值转换为标准单位

输入格式：包含entities数组的JSON
输出格式：在每个entity中添加standard_code、standard_name、standard_system字段

只返回JSON。"""

            self.std_agent = ReActAgent(
                name="CSVStandardizer",
                sys_prompt=standardization_prompt,
                model=self.model,
                formatter=OpenAIChatFormatter(),
            )
            
            # 设置提取模式的schema
            try:
                from schema import MedicalExtractionResult
                self.extraction_schema = MedicalExtractionResult
            except ImportError:
                self.extraction_schema = None
            
            if self.verbose:
                print("[CSV处理器] 信息抽取Agent初始化成功")
                
        except Exception as e:
            if self.verbose:
                print(f"[CSV处理器] 信息抽取Agent初始化失败: {e}")
            self.med_agent = None
            self.std_agent = None
    
    async def process_file(
        self,
        file_path: str,
        standardization_columns: Optional[List[str]] = None,
        value_columns: Optional[List[str]] = None,
        unit_columns: Optional[Dict[str, str]] = None,
        category_mapping: Optional[Dict[str, str]] = None,
        extraction_columns: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理CSV文件
        
        Args:
            file_path: CSV文件路径
            standardization_columns: 需要术语标准化的列名列表（已结构化的列）
                                    如果为None，将自动分析
            value_columns: 需要量纲统一的数值列名列表
            unit_columns: 数值列对应的单位列映射 {value_col: unit_col}
            category_mapping: 列名到医学类别的映射 {col_name: category}
                             category可以是 Disease, Drug, Test, Symptom等
            extraction_columns: 需要进行信息抽取的文本列名列表
                               这些列包含自由文本（如放射报告、出院记录等），
                               将先进行信息抽取，然后对提取的实体进行标准化
            max_rows: 最大处理行数
            output_path: 输出文件路径
            
        Returns:
            处理结果字典
        """
        result = {
            "success": False,
            "file_path": file_path,
            "processed_at": datetime.now().isoformat(),
            "statistics": {},
            "data": [],
            "extracted_entities": [],  # 从文本列抽取的实体
            "error": None
        }
        
        # 1. 读取并分析CSV文件
        if self.verbose:
            print(f"[CSV处理器] 开始处理文件: {file_path}")
        
        sample_result = read_csv_sample(file_path, sample_rows=5)
        if not sample_result["success"]:
            result["error"] = sample_result["error"]
            return result
        
        columns = sample_result["columns"]
        
        # 2. 自动分析需要处理的列（如果未指定）
        analysis = analyze_columns_for_standardization(columns, sample_result["sample_data"])
        
        if standardization_columns is None:
            standardization_columns = [
                item["column"] for item in analysis["standardization_candidates"]
            ]
        
        if value_columns is None:
            value_columns = [
                item["column"] for item in analysis["unit_normalization_candidates"]
            ]
        
        if unit_columns is None:
            # 尝试自动匹配单位列
            unit_columns = self._auto_match_unit_columns(
                columns, 
                value_columns,
                analysis.get("unit_columns", [])
            )
        
        # 自动检测需要信息抽取的文本列
        if extraction_columns is None:
            extraction_columns = self._detect_extraction_columns(columns, sample_result["sample_data"])
        
        if self.verbose:
            print(f"  - 标准化列: {standardization_columns}")
            print(f"  - 量纲统一列: {value_columns}")
            print(f"  - 单位列映射: {unit_columns}")
            print(f"  - 信息抽取列: {extraction_columns}")
        
        # 3. 读取全部数据
        csv_data = read_csv_data(file_path, max_rows=max_rows)
        if not csv_data["success"]:
            result["error"] = csv_data["error"]
            return result
        
        result["statistics"]["total_rows"] = csv_data["row_count"]
        result["statistics"]["standardization_columns"] = standardization_columns
        result["statistics"]["value_columns"] = value_columns
        result["statistics"]["extraction_columns"] = extraction_columns
        
        # 4. 处理每一行数据
        processed_data = []
        all_extracted_entities = []
        stats = {
            "terms_standardized": 0,
            "terms_not_found": 0,
            "values_normalized": 0,
            "values_unchanged": 0,
            "texts_extracted": 0,
            "entities_extracted": 0
        }
        
        for row_idx, row in enumerate(csv_data["data"]):
            if self.verbose and row_idx % 100 == 0:
                print(f"  处理进度: {row_idx}/{csv_data['row_count']}")
            
            processed_row = row.copy()
            row_entities = []
            
            # 4.1 对文本列进行信息抽取
            for col in extraction_columns:
                if col in row and row[col] and len(str(row[col]).strip()) > 50:
                    text_content = str(row[col]).strip()
                    
                    # 进行信息抽取
                    extraction_result = await self._extract_from_text(text_content, row_idx, col)
                    
                    if extraction_result and extraction_result.get("entities"):
                        entities = extraction_result["entities"]
                        stats["texts_extracted"] += 1
                        stats["entities_extracted"] += len(entities)
                        
                        # 保存抽取结果到该行
                        processed_row[f"{col}_entities"] = json.dumps(entities, ensure_ascii=False)
                        processed_row[f"{col}_entity_count"] = len(entities)
                        
                        # 保存印象/结论
                        if extraction_result.get("impression"):
                            processed_row[f"{col}_impression"] = extraction_result["impression"]
                        if extraction_result.get("indication"):
                            processed_row[f"{col}_indication"] = extraction_result["indication"]
                        
                        # 收集实体用于标准化
                        for entity in entities:
                            entity["source_row"] = row_idx
                            entity["source_column"] = col
                            row_entities.append(entity)
                        
                        # 对抽取的实体进行标准化
                        for entity in entities:
                            category = entity.get("category", "Finding")
                            name = entity.get("name", "")
                            if name:
                                std_result = await self._standardize_term(name, category)
                                if std_result and std_result.get("code"):
                                    entity["standard_code"] = std_result["code"]
                                    entity["standard_name"] = std_result.get("name", "")
                                    entity["standard_system"] = std_result.get("system", "")
                                    stats["terms_standardized"] += 1
                                else:
                                    stats["terms_not_found"] += 1
                        
                        # 更新已标准化的实体
                        processed_row[f"{col}_standardized_entities"] = json.dumps(entities, ensure_ascii=False)
            
            # 4.2 术语标准化（已结构化的列）
            for col in standardization_columns:
                if col in row and row[col]:
                    # 确定类别
                    category = "Test"  # 默认类别
                    if category_mapping and col in category_mapping:
                        category = category_mapping[col]
                    
                    std_result = await self._standardize_term(
                        row[col], category
                    )
                    
                    if std_result and std_result.get("code"):
                        processed_row[f"{col}_standard_code"] = std_result["code"]
                        processed_row[f"{col}_standard_name"] = std_result.get("name", "")
                        processed_row[f"{col}_standard_system"] = std_result.get("system", "")
                        stats["terms_standardized"] += 1
                    else:
                        stats["terms_not_found"] += 1
            
            # 4.3 量纲统一
            for val_col in value_columns:
                if val_col in row and row[val_col]:
                    try:
                        value = float(row[val_col])
                        
                        # 获取单位
                        unit = None
                        if unit_columns and val_col in unit_columns:
                            unit_col = unit_columns[val_col]
                            unit = row.get(unit_col, "")
                        
                        if unit:
                            # 获取检验名称（用于量纲统一）
                            test_name = self._get_test_name_for_value(row, val_col, standardization_columns)
                            
                            norm_result = await self._normalize_unit(
                                test_name, value, unit
                            )
                            
                            if norm_result[1] != unit:
                                processed_row[f"{val_col}_normalized"] = norm_result[0]
                                processed_row[f"{val_col}_normalized_unit"] = norm_result[1]
                                stats["values_normalized"] += 1
                            else:
                                stats["values_unchanged"] += 1
                        else:
                            stats["values_unchanged"] += 1
                            
                    except (ValueError, TypeError):
                        stats["values_unchanged"] += 1
            
            processed_data.append(processed_row)
            all_extracted_entities.extend(row_entities)
        
        result["data"] = processed_data
        result["extracted_entities"] = all_extracted_entities
        result["statistics"].update(stats)
        result["success"] = True
        
        # 5. 保存JSON输出（如果指定）
        if output_path:
            self._save_output(output_path, processed_data)
            result["output_path"] = output_path
            if self.verbose:
                print(f"  JSON结果已保存到: {output_path}")
        
        # 6. 自动生成结构化CSV结果
        from pathlib import Path
        input_file = Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确定结果目录
        results_dir = input_file.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        csv_output_path = results_dir / f"{input_file.stem}_processed_{timestamp}.csv"
        
        # 保存结构化CSV
        saved_csv = self.save_as_structured_csv(result, str(csv_output_path))
        if saved_csv:
            result["csv_output_path"] = saved_csv
        
        if self.verbose:
            print(f"[CSV处理器] 处理完成: {stats}")
            if stats["texts_extracted"] > 0:
                print(f"  - 文本抽取: {stats['texts_extracted']} 个文本处理")
                print(f"  - 实体抽取: {stats['entities_extracted']} 个实体提取")
        
        return result
    
    async def _standardize_term(self, term: str, category: str) -> Dict:
        """标准化术语"""
        if self._async_mode:
            return await self.standardize_term(
                term, category,
                use_cache=True,
                model=self.model,
                fallback_to_local=True
            )
        else:
            return self.standardize_term(
                term, category,
                use_umls=self.use_umls
            )
    
    async def _normalize_unit(self, test_name: str, value: float, unit: str) -> Tuple:
        """量纲统一"""
        if self._async_mode:
            result = await self.normalize_unit(
                test_name, value, unit,
                model=self.model,
                use_umls=self.use_umls
            )
            # 返回 (value, unit) 或 (value, unit, source)
            return result[:2]
        else:
            return self.normalize_unit(test_name, value, unit, use_umls=self.use_umls)
    
    def _detect_extraction_columns(
        self,
        columns: List[str],
        sample_data: List[Dict]
    ) -> List[str]:
        """
        自动检测需要信息抽取的文本列
        
        识别包含长文本（如放射报告、出院记录等）的列
        
        Args:
            columns: 所有列名
            sample_data: 样本数据
            
        Returns:
            需要信息抽取的列名列表
        """
        extraction_columns = []
        
        # 可能包含自由文本的列名关键词
        text_keywords = [
            "text", "note", "report", "description", "comment", "narrative",
            "findings", "impression", "indication", "conclusion", "summary",
            "discharge", "radiology", "pathology", "history", "assessment"
        ]
        
        for col in columns:
            col_lower = col.lower()
            
            # 检查列名是否包含文本相关关键词
            is_text_column = any(kw in col_lower for kw in text_keywords)
            
            # 检查样本数据中该列的内容长度
            has_long_text = False
            for row in sample_data:
                if col in row and row[col]:
                    text_len = len(str(row[col]).strip())
                    # 如果文本长度超过100个字符，认为是长文本
                    if text_len > 100:
                        has_long_text = True
                        break
            
            # 如果列名符合且有长文本，添加到抽取列表
            if is_text_column and has_long_text:
                extraction_columns.append(col)
        
        return extraction_columns
    
    async def _extract_from_text(
        self,
        text: str,
        row_idx: int,
        column_name: str
    ) -> Optional[Dict]:
        """
        从文本中提取医学实体
        
        Args:
            text: 要处理的文本
            row_idx: 行索引
            column_name: 列名
            
        Returns:
            抽取结果，包含entities列表
        """
        if not self.use_llm:
            return None
        
        # 如果Agent未初始化，尝试使用简单的LLM调用
        if not self.med_agent:
            return await self._simple_extract(text)
        
        try:
            from agentscope.message import Msg
            
            # 构造消息
            msg = Msg(
                name="User",
                content=f"Extract structured information from the following medical text. IMPORTANT: Keep the SAME language as the input (do NOT translate).\n\n{text[:4000]}",  # 限制长度
                role="user"
            )
            
            # 调用Agent
            if self.extraction_schema:
                res = await self.med_agent(msg, structured_model=self.extraction_schema)
            else:
                res = await self.med_agent(msg)
            
            if res.metadata:
                return res.metadata
            elif res.content:
                # 尝试解析JSON响应
                return self._parse_extraction_response(res.content)
                
        except Exception as e:
            if self.verbose:
                print(f"  [行{row_idx}][{column_name}] 信息抽取失败: {e}")
        
        return None
    
    async def _simple_extract(self, text: str) -> Optional[Dict]:
        """
        使用简单的LLM调用进行信息抽取（不使用Agent框架）
        
        Args:
            text: 要处理的文本
            
        Returns:
            抽取结果
        """
        try:
            import openai
            from config.settings import get_api_config
            
            config = get_api_config()
            if not config.get('api_key'):
                return None
            
            client = openai.AsyncOpenAI(
                api_key=config['api_key'],
                base_url=config['api_base']
            )
            
            prompt = f"""Extract **concise key-value pairs** of structured information from the following medical text.

## CRITICAL: Language Rule
- **MUST keep the SAME language as the input text**
- If input is English, output in English
- If input is Chinese, output in Chinese
- NEVER translate the content

Medical text:
{text[:4000]}

## Extraction Requirements
1. **Concise**: Keep name and value short (typically 2-6 words)
2. **Split**: Break complex descriptions into multiple short entities
3. **Format**: "location/item: finding/result"

## Categories
- Finding: name=location, value=finding result, e.g., {{"name":"Lung","value":"no consolidation"}} or {{"name":"肝脏","value":"回声增粗"}}
- Disease: name=disease name, value=severity
- Test: name=test name
- Anatomy: name=location

## Examples

### English input:
"No focal consolidation. No pleural effusion."
Output:
{{"entities":[
    {{"category":"Finding","name":"Lung","value":"no focal consolidation"}},
    {{"category":"Finding","name":"Pleural space","value":"no effusion"}}
]}}

### Chinese input:
"肝脏回声增粗，未见占位。少量腹水。"
Output:
{{"entities":[
    {{"category":"Finding","name":"肝脏","value":"回声增粗"}},
    {{"category":"Finding","name":"肝脏占位","value":"未见"}},
    {{"category":"Finding","name":"腹腔","value":"少量腹水"}}
]}}

Return JSON:
{{
    "entities": [{{"category":"Category","name":"short_name","value":"short_result"}}],
    "impression": "Overall impression (one sentence, same language as input)",
    "indication": "Indication (if any)"
}}

Return JSON only."""
            
            response = await client.chat.completions.create(
                model=config['model_name'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_extraction_response(content)
            
        except Exception as e:
            if self.verbose:
                print(f"  简单抽取失败: {e}")
            return None
    
    def _parse_extraction_response(self, content: str) -> Optional[Dict]:
        """
        解析LLM的抽取响应
        
        Args:
            content: LLM响应内容
            
        Returns:
            解析后的字典
        """
        try:
            # 尝试提取JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            # 确保有entities字段
            if "entities" not in result:
                result["entities"] = []
            
            return result
            
        except json.JSONDecodeError:
            return None
    
    def _auto_match_unit_columns(
        self,
        columns: List[str],
        value_columns: List[str],
        detected_unit_columns: List[Dict]
    ) -> Dict[str, str]:
        """
        自动匹配数值列和单位列
        
        Args:
            columns: 所有列名
            value_columns: 数值列
            detected_unit_columns: 检测到的单位列
            
        Returns:
            映射字典 {value_col: unit_col}
        """
        mapping = {}
        unit_col_names = [item["column"] for item in detected_unit_columns]
        
        # 简单启发式：查找名字相近的列
        for val_col in value_columns:
            for unit_col in unit_col_names:
                # 检查是否是同一组（如 dilution_value 和 dilution_text）
                val_base = val_col.lower().replace("_value", "").replace("_result", "")
                unit_base = unit_col.lower().replace("_text", "").replace("_unit", "")
                
                if val_base in unit_base or unit_base in val_base:
                    mapping[val_col] = unit_col
                    break
        
        return mapping
    
    def _get_test_name_for_value(
        self,
        row: Dict,
        value_column: str,
        standardization_columns: List[str]
    ) -> str:
        """
        获取数值对应的检验名称
        
        Args:
            row: 当前行数据
            value_column: 数值列名
            standardization_columns: 标准化列
            
        Returns:
            检验名称
        """
        # 尝试从标准化列中获取检验名称
        for col in standardization_columns:
            if col in row and row[col]:
                return row[col]
        
        # 如果没有，使用列名作为名称
        return value_column
    
    def _save_output(self, output_path: str, data: List[Dict]) -> None:
        """保存处理结果"""
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            # 保存为CSV
            if data:
                fieldnames = list(data[0].keys())
                with open(output_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
    
    def _parse_value_and_unit(self, text: str) -> Tuple[str, str]:
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
        # 匹配模式：数字（可带小数）+ 可选空格 + 单位
        patterns = [
            r'^([-+]?\d*\.?\d+)\s*(.*)$',  # 数字开头
            r'^([<>≤≥]?\s*[-+]?\d*\.?\d+)\s*(.*)$',  # 可能带比较符号
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                value_part = match.group(1).strip()
                unit_part = match.group(2).strip()
                if value_part:  # 确保有数值部分
                    return value_part, unit_part
        
        # 无法分离，返回原文作为值，单位为空
        return text, ""
    
    def save_as_structured_csv(
        self,
        result: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        将处理结果保存为结构化的CSV文件
        
        保留原始列，添加处理结果列：
        - 对于抽取列（extraction_columns）：
          - {col}_印象: 检查印象/结论
          - {col}_指征: 检查指征
          - {col}_{Category}_抽取: 按类别汇总的键值对（保留原格式）
          - {col}_{Category}_标准化: 按类别汇总的标准化结果
          - Extracted_{Category}_{EntityName}: 以实体名称为列名，值为对应value
          - Extracted_{Category}_{EntityName}_value: 数值部分
          - Extracted_{Category}_{EntityName}_unit: 单位部分
        - 对于标准化列（standardization_columns）：
          - {col}_标准化: 标准化结果
        - 对于量纲统一列（value_columns）：
          - {col}_量纲统一_value: 统一后的数值
          - {col}_量纲统一_unit: 统一后的单位
        
        所有字段都会保存，即使为空。
        
        Args:
            result: 处理结果字典
            output_path: 输出CSV路径
            
        Returns:
            保存的文件路径
        """
        # 实体类别
        ENTITY_CATEGORIES = ["Test", "Disease", "Drug", "Symptom", "Treatment", "Anatomy", "Finding", "Other"]
        
        processed_data = result.get("data", [])
        extraction_columns = result.get("statistics", {}).get("extraction_columns", [])
        standardization_columns = result.get("statistics", {}).get("standardization_columns", [])
        value_columns = result.get("statistics", {}).get("value_columns", [])
        
        if not processed_data:
            return ""
        
        # 第一遍：收集所有抽取的实体名称，用于生成固定列
        all_entity_keys = set()  # 格式：(column, category, entity_name)
        
        for row in processed_data:
            for col in extraction_columns:
                entities_json = row.get(f"{col}_standardized_entities", "[]")
                try:
                    entities = json.loads(entities_json) if isinstance(entities_json, str) else (entities_json or [])
                except json.JSONDecodeError:
                    entities = []
                
                for entity in entities:
                    cat = entity.get("category", "Other")
                    if cat not in ENTITY_CATEGORIES:
                        cat = "Other"
                    name = entity.get("name", "").strip()
                    if name:
                        # 清理实体名称，使其适合作为列名
                        clean_name = name.replace("/", "-").replace("\\", "-").replace(":", "-")
                        all_entity_keys.add((col, cat, clean_name))
        
        # 按 (column, category, name) 排序
        sorted_entity_keys = sorted(all_entity_keys, key=lambda x: (x[0], ENTITY_CATEGORIES.index(x[1]) if x[1] in ENTITY_CATEGORIES else 999, x[2]))
        
        # 收集所有行数据
        rows = []
        
        for row_idx, row in enumerate(processed_data):
            new_row = {}
            
            # 1. 复制原始列（排除处理中添加的中间列）
            skip_suffixes = [
                '_entities', '_entity_count', '_standardized_entities',
                '_impression', '_indication', '_standard_code', 
                '_standard_name', '_standard_system', '_normalized', '_normalized_unit'
            ]
            for key, value in row.items():
                if not any(key.endswith(suffix) for suffix in skip_suffixes):
                    new_row[key] = value
            
            # 2. 处理抽取列的结果
            for col in extraction_columns:
                # 印象和指征 - 始终保存（即使为空）
                new_row[f"{col}_印象"] = row.get(f"{col}_impression", "")
                new_row[f"{col}_指征"] = row.get(f"{col}_indication", "")
                
                # 获取标准化后的实体
                entities_json = row.get(f"{col}_standardized_entities", "[]")
                try:
                    entities = json.loads(entities_json) if isinstance(entities_json, str) else (entities_json or [])
                except json.JSONDecodeError:
                    entities = []
                
                # ========== 按类别汇总（保留原格式） ==========
                by_category = {cat: {"抽取": [], "标准化": []} for cat in ENTITY_CATEGORIES}
                
                # 构建实体查找字典，同时按类别汇总
                entity_dict = {}  # key: (category, name), value: entity
                for entity in entities:
                    cat = entity.get("category", "Other")
                    if cat not in ENTITY_CATEGORIES:
                        cat = "Other"
                    name = entity.get("name", "").strip()
                    value = entity.get("value", "")
                    unit = entity.get("unit", "")
                    
                    if name:
                        clean_name = name.replace("/", "-").replace("\\", "-").replace(":", "-")
                        entity_dict[(cat, clean_name)] = entity
                        
                        # 按类别汇总 - 键值对形式 "名称:值"
                        if value:
                            entity_str = f"{name}:{value}"
                            if unit:
                                entity_str += f" {unit}"
                        else:
                            entity_str = name
                        by_category[cat]["抽取"].append(entity_str)
                        
                        # 标准化结果汇总
                        std_code = entity.get("standard_code", "")
                        std_name = entity.get("standard_name", "")
                        std_system = entity.get("standard_system", "")
                        
                        if std_code and std_name and std_name != name:
                            std_str = f"{name}→{std_name}"
                            if std_system:
                                std_str += f"[{std_system}]"
                            by_category[cat]["标准化"].append(std_str)
                
                # 保存按类别汇总的结果（始终保存所有类别列，即使为空）
                for cat in ENTITY_CATEGORIES:
                    new_row[f"{col}_{cat}_抽取"] = "; ".join(by_category[cat]["抽取"]) if by_category[cat]["抽取"] else ""
                    new_row[f"{col}_{cat}_标准化"] = "; ".join(by_category[cat]["标准化"]) if by_category[cat]["标准化"] else ""
                
                # ========== 按实体名称展开为独立列 ==========
                for ext_col, cat, entity_name in sorted_entity_keys:
                    if ext_col != col:
                        continue
                    
                    # 列名前缀：Extracted_{Category}_{EntityName}
                    col_prefix = f"Extracted_{cat}_{entity_name}"
                    
                    entity = entity_dict.get((cat, entity_name))
                    
                    if entity:
                        # 获取实体的值
                        value = entity.get("value", "")
                        unit = entity.get("unit", "")
                        
                        # 组合原始抽取结果（只是value，不含name）
                        if value:
                            entity_str = f"{value}"
                            if unit:
                                entity_str += f" {unit}"
                        else:
                            entity_str = ""
                        
                        # 分离数值和单位
                        parsed_value, parsed_unit = self._parse_value_and_unit(entity_str)
                        
                        new_row[col_prefix] = entity_str
                        new_row[f"{col_prefix}_value"] = parsed_value
                        new_row[f"{col_prefix}_unit"] = parsed_unit
                    else:
                        # 本行没有该实体，填充空值
                        new_row[col_prefix] = ""
                        new_row[f"{col_prefix}_value"] = ""
                        new_row[f"{col_prefix}_unit"] = ""
            
            # 3. 处理直接标准化列的结果（始终保存列，即使为空）
            for col in standardization_columns:
                original = row.get(col, "")
                std_code = row.get(f"{col}_standard_code", "")
                std_name = row.get(f"{col}_standard_name", "")
                std_system = row.get(f"{col}_standard_system", "")
                
                if std_code and std_name and std_name != original:
                    new_row[f"{col}_标准化"] = f"{original} → {std_name} [{std_system}:{std_code}]"
                else:
                    new_row[f"{col}_标准化"] = ""
            
            # 4. 处理量纲统一列的结果（拆分数值和单位，始终保存）
            for col in value_columns:
                original_value = row.get(col, "")
                normalized_value = row.get(f"{col}_normalized")
                normalized_unit = row.get(f"{col}_normalized_unit", "")
                
                # 获取原始单位
                original_unit = ""
                unit_columns_mapping = result.get("statistics", {}).get("unit_columns", {})
                if col in unit_columns_mapping:
                    original_unit = row.get(unit_columns_mapping[col], "")
                
                # 始终保存这些列
                if normalized_value is not None and str(normalized_value) != str(original_value):
                    new_row[f"{col}_量纲统一_value"] = normalized_value
                    new_row[f"{col}_量纲统一_unit"] = normalized_unit
                    new_row[f"{col}_量纲统一_原值"] = f"{original_value} {original_unit}".strip()
                else:
                    new_row[f"{col}_量纲统一_value"] = ""
                    new_row[f"{col}_量纲统一_unit"] = ""
                    new_row[f"{col}_量纲统一_原值"] = ""
            
            rows.append(new_row)
        
        if not rows:
            return ""
        
        # 获取原始列顺序
        original_columns = []
        if processed_data:
            for key in processed_data[0].keys():
                if not any(key.endswith(s) for s in [
                    '_entities', '_entity_count', '_standardized_entities',
                    '_impression', '_indication', '_standard_code', 
                    '_standard_name', '_standard_system', '_normalized', '_normalized_unit'
                ]):
                    original_columns.append(key)
        
        # 组织列顺序：原始列 -> 印象/指征 -> 按类别汇总 -> 按实体展开 -> 标准化列结果 -> 量纲统一结果
        columns = []
        
        # 原始列
        for col in original_columns:
            columns.append(col)
        
        # 抽取列结果：先印象和指征
        for ext_col in extraction_columns:
            columns.append(f"{ext_col}_印象")
            columns.append(f"{ext_col}_指征")
            
            # 按类别汇总的列
            for cat in ENTITY_CATEGORIES:
                columns.append(f"{ext_col}_{cat}_抽取")
                columns.append(f"{ext_col}_{cat}_标准化")
        
        # 按实体展开的列
        for ext_col, cat, entity_name in sorted_entity_keys:
            col_prefix = f"Extracted_{cat}_{entity_name}"
            columns.append(col_prefix)
            columns.append(f"{col_prefix}_value")
            columns.append(f"{col_prefix}_unit")
        
        # 标准化列结果
        for std_col in standardization_columns:
            columns.append(f"{std_col}_标准化")
        
        # 量纲统一列结果
        for val_col in value_columns:
            columns.append(f"{val_col}_量纲统一_原值")
            columns.append(f"{val_col}_量纲统一_value")
            columns.append(f"{val_col}_量纲统一_unit")
        
        # 去重并保持顺序
        seen = set()
        unique_columns = []
        for col in columns:
            if col not in seen:
                seen.add(col)
                unique_columns.append(col)
        columns = unique_columns
        
        # 写入CSV
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                # 确保所有列都有值（即使为空）
                complete_row = {col: row.get(col, "") for col in columns}
                writer.writerow(complete_row)
        
        if self.verbose:
            print(f"✅ 结构化CSV结果已保存: {output_path}")
            print(f"   - 总行数: {len(rows)}")
            print(f"   - 列数: {len(columns)}")
            print(f"   - 抽取实体数: {len(sorted_entity_keys)}")
        
        return output_path


async def process_csv_file(
    file_path: str,
    standardization_columns: Optional[List[str]] = None,
    value_columns: Optional[List[str]] = None,
    unit_columns: Optional[Dict[str, str]] = None,
    category_mapping: Optional[Dict[str, str]] = None,
    extraction_columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    output_path: Optional[str] = None,
    use_llm: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    处理CSV文件的便捷函数
    
    Args:
        file_path: CSV文件路径
        standardization_columns: 需要术语标准化的列（已结构化的列）
        value_columns: 需要量纲统一的数值列
        unit_columns: 单位列映射
        category_mapping: 类别映射
        extraction_columns: 需要信息抽取的文本列（如放射报告、出院记录等）
        max_rows: 最大处理行数
        output_path: 输出路径
        use_llm: 是否使用LLM
        verbose: 是否详细输出
        
    Returns:
        处理结果
    """
    processor = CSVProcessor(use_llm=use_llm, verbose=verbose)
    return await processor.process_file(
        file_path=file_path,
        standardization_columns=standardization_columns,
        value_columns=value_columns,
        unit_columns=unit_columns,
        category_mapping=category_mapping,
        extraction_columns=extraction_columns,
        max_rows=max_rows,
        output_path=output_path
    )
