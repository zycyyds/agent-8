# -*- coding: utf-8 -*-
"""
统一数据处理Agent

根据数据类型自动选择合适的处理流程：
- 图片：OCR -> 预处理 -> 信息抽取 -> 标准化 -> 量纲统一
- 文本：预处理 -> 信息抽取 -> 标准化 -> 量纲统一
- CSV：标准化 -> 量纲统一（跳过预处理和信息抽取）
"""
import os
import sys
import json
from typing import Dict, Optional, Any, List
from datetime import datetime

# 设置导入路径
module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

parent_dir = os.path.dirname(module_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

src_path = os.path.abspath(os.path.join(parent_dir, "../../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from agentscope.agent import AgentBase, ReActAgent
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.tool import Toolkit

from config.settings import DataType, ProcessingStage, get_api_config
from tools.data_type_detector import detect_data_type
from tools.csv_reader import get_csv_info_for_agent
from tools.folder_analyzer import (
    analyze_folder_for_agent, 
    group_files_by_patient,
    get_folder_summary
)
from processors.csv_processor import CSVProcessor


class UnifiedProcessingAgent(AgentBase):
    """
    统一数据处理Agent
    
    智能识别输入数据类型，并自动选择合适的处理流程。
    支持处理图片、文本和CSV三种类型的医学数据。
    """
    
    def __init__(
        self,
        name: str = "UnifiedProcessor",
        model: Optional[OpenAIChatModel] = None,
        use_llm: bool = True,
        use_umls: bool = True,
        verbose: bool = False
    ):
        """
        初始化统一处理Agent
        
        Args:
            name: Agent名称
            model: LLM模型实例
            use_llm: 是否使用LLM进行智能分析和标准化
            use_umls: 是否使用UMLS API
            verbose: 是否打印详细信息
        """
        super().__init__()
        self.name = name
        self.use_llm = use_llm
        self.use_umls = use_umls
        self.verbose = verbose
        
        # 初始化模型
        if model:
            self.model = model
        elif use_llm:
            self._init_model()
        else:
            self.model = None
        
        # 初始化子处理器
        self._init_processors()
        
        # 处理统计
        self.processing_stats = {
            "total_processed": 0,
            "by_type": {
                "image": 0,
                "text": 0,
                "csv": 0,
            },
            "errors": 0
        }
    
    def _init_model(self):
        """初始化LLM模型"""
        config = get_api_config()
        if config['api_key']:
            try:
                self.model = OpenAIChatModel(
                    config_name="unified-processor-model",
                    model_name=config['model_name'],
                    api_key=config['api_key'],
                    client_kwargs={
                        "base_url": config['api_base'],
                        "timeout": config['timeout'],
                    },
                    generate_kwargs={"temperature": 0.0}
                )
            except Exception as e:
                if self.verbose:
                    print(f"[{self.name}] 初始化模型失败: {e}")
                self.model = None
        else:
            self.model = None
    
    def _init_processors(self):
        """初始化各类型的处理器"""
        # CSV处理器
        self.csv_processor = CSVProcessor(
            use_llm=self.use_llm,
            use_umls=self.use_umls,
            model=self.model,
            verbose=self.verbose
        )
        
        # 图片/文本处理器（复用原有的Agent）
        self._init_image_text_agents()
    
    def _init_image_text_agents(self):
        """初始化图片和文本处理的Agent"""
        if not self.model:
            self.med_agent = None
            self.std_agent = None
            return
        
        try:
            # 导入原有的工具和Agent
            try:
                from tools.tools_preprocess import preprocess_medical_input
                from tools.tools_ocr import extract_text_from_image, is_image_file
            except ImportError:
                from tools.tools_preprocess import preprocess_medical_input
                from tools.tools_ocr import extract_text_from_image, is_image_file
            from agentscope.tool._text_processing._medical_clean import clean_medical_text
            try:
                from schema import MedicalExtractionResult  # 从项目根目录导入
            except ImportError:
                from ..schema import MedicalExtractionResult
            
            # 创建工具包
            toolkit = Toolkit()
            toolkit.create_tool_group(
                group_name="medical",
                description="Medical data processing tools",
                active=True,
            )
            
            toolkit.register_tool_function(
                preprocess_medical_input,
                group_name="medical",
                func_description="Preprocess medical input (text or image file path)."
            )
            toolkit.register_tool_function(
                extract_text_from_image,
                group_name="medical",
                func_description="Extract text from a medical image file using OCR."
            )
            toolkit.register_tool_function(
                clean_medical_text,
                group_name="medical",
                func_description="Clean medical text by normalizing symbols and whitespace."
            )
            
            # 创建医学信息提取Agent
            self.med_agent = ReActAgent(
                name="MedicalExpert",
                sys_prompt="""你是一位专业的医学数据提取专家，从各类医学文档（检验报告、病历、处方、诊断书等）中提取结构化信息。

## 工作流程
1. 如果输入是图片路径，先使用 `preprocess_medical_input` 工具进行OCR识别
2. 如果输入是文本，可以使用 `clean_medical_text` 工具清洗
3. 仔细分析文本，提取所有医学实体

## 实体类别及提取要求

### 1. Test（检验项目）
- **name**: 使用完整标准名称（如"类风湿因子"而非"风湿因子"，"白细胞计数"而非"白细胞"）
- **value**: 数值保持原始精度，定性结果用字符串（如"阴性"）
- **unit**: 标准格式（10^9/L、IU/mL、g/L、mg/dL、%等）

### 2. Disease（疾病/诊断）
- **name**: 使用标准疾病名称（如"2型糖尿病"而非"糖尿病"，"原发性高血压"而非"高血压"）
- **value**: 可填写分期/分级（如"III期"、"中度"）

### 3. Drug（药物）
- **name**: 使用通用名或商品名（如"阿司匹林"、"拜阿司匹灵"）
- **value**: 剂量（如"100mg"）
- **unit**: 用法（如"每日一次"、"口服"）

### 4. Symptom（症状/主诉）
- **name**: 症状名称（如"头痛"、"恶心"、"发热"）
- **value**: 程度或持续时间（如"剧烈"、"3天"）

### 5. Treatment（治疗/手术）
- **name**: 治疗名称（如"冠状动脉搭桥术"、"化疗"）
- **value**: 相关信息（如"第3周期"）

### 6. Anatomy（解剖部位）
- **name**: 部位名称（如"左侧颞叶"、"右肺下叶"）

### 7. LabValue（实验室数值）- 用于其他定量指标
- **name**: 指标名称
- **value**: 数值
- **unit**: 单位

## 重要：时间信息（temporal_info）
**时间信息不要放在 entities 中！** 放在 temporal_info 列表中。
格式：{"event": "检查时间", "time_expression": "2021-03-01", "normalized_time": "2021-03-01"}

## 重要：剂量信息（quantity_info）
**剂量信息不要放在 entities 中！** 放在 quantity_info 列表中。
格式：{"drug_or_treatment": "阿司匹林", "amount": "100mg", "frequency": "每日一次"}

## 关系（relations）
提取实体间关系：
- TreatmentFor: 药物/治疗 → 疾病
- ManifestationOf: 症状 → 疾病  
- TestResult: 检验 → 异常/正常
- Causal: 原因 → 结果

## 输出要求
- **entities只能是以下category之一：Disease, Drug, Symptom, Test, Treatment, Anatomy, LabValue, Finding, Other**
- **绝对不要在entities中使用 Temporal、Time、Date 等category**
- 每个entity必须有 name、category、original_text
- 尽量纠正OCR错误（如"美风湿因子"→"类风湿因子"）
- 不要遗漏任何有临床意义的信息""",
                model=self.model,
                formatter=OpenAIChatFormatter(),
                toolkit=toolkit,
            )
            
            # 创建标准化Agent
            try:
                from agents.standardization_agent_llm import EnhancedMedicalStandardizationAgent
                self.std_agent = EnhancedMedicalStandardizationAgent(
                    name="Standardizer",
                    use_llm=True,
                    use_cache=True,
                    model=self.model
                )
            except ImportError:
                from agents.standardization_agent import MedicalStandardizationAgent
                self.std_agent = MedicalStandardizationAgent(name="Standardizer")
            
            self.extraction_schema = MedicalExtractionResult
            
        except Exception as e:
            if self.verbose:
                print(f"[{self.name}] 初始化图片/文本处理Agent失败: {e}")
            self.med_agent = None
            self.std_agent = None
    
    async def reply(self, msg: Msg | dict | None = None) -> Msg:
        """
        处理输入消息
        
        Args:
            msg: 输入消息，包含文件路径或目录路径
            
        Returns:
            处理结果消息
        """
        if msg is None:
            return Msg(self.name, "未提供输入", role="assistant")
        
        # 解析输入
        if isinstance(msg, Msg):
            input_path = msg.content
        elif isinstance(msg, dict):
            input_path = msg.get("content", "")
        else:
            input_path = str(msg)
        
        if self.verbose:
            print(f"\n[{self.name}] 开始处理: {input_path}")
        
        # 检查是否是目录
        if os.path.isdir(input_path):
            return await self._process_directory(input_path)
        
        # 1. 检测数据类型
        data_type, metadata = detect_data_type(input_path)
        
        if self.verbose:
            print(f"[{self.name}] 检测到数据类型: {data_type.value}")
        
        # 2. 根据类型选择处理流程
        try:
            if data_type == DataType.CSV:
                result = await self._process_csv(input_path)
            elif data_type == DataType.IMAGE:
                result = await self._process_image(input_path)
            elif data_type == DataType.TEXT:
                result = await self._process_text(input_path)
            else:
                result = {
                    "success": False,
                    "error": f"不支持的数据类型: {data_type.value}",
                    "data_type": data_type.value
                }
            
            # 更新统计
            self.processing_stats["total_processed"] += 1
            if data_type.value in self.processing_stats["by_type"]:
                self.processing_stats["by_type"][data_type.value] += 1
            
            if not result.get("success"):
                self.processing_stats["errors"] += 1
            
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "data_type": data_type.value
            }
            self.processing_stats["errors"] += 1
            if self.verbose:
                print(f"[{self.name}] 处理出错: {e}")
        
        # 3. 返回结果
        description = self._format_result(result)
        
        return Msg(
            self.name,
            content=description,
            role="assistant",
            metadata=result
        )
    
    async def _process_directory(self, dir_path: str) -> Msg:
        """
        处理目录 - 分析结构并按病人分组处理
        
        Args:
            dir_path: 目录路径
            
        Returns:
            处理结果消息
        """
        if self.verbose:
            print(f"\n[{self.name}] 分析目录结构: {dir_path}")
            print(get_folder_summary(dir_path))
        
        # 分析目录结构
        analysis = analyze_folder_for_agent(dir_path)
        
        if "error" in analysis:
            return Msg(
                self.name,
                content=f"目录分析失败: {analysis['error']}",
                role="assistant",
                metadata={"success": False, "error": analysis['error']}
            )
        
        # 获取按病人分组的文件
        patient_groups = analysis.get("patients", {})
        
        if not patient_groups:
            return Msg(
                self.name,
                content="目录中没有找到可处理的图片文件",
                role="assistant",
                metadata={"success": False, "error": "no_files_found"}
            )
        
        if self.verbose:
            print(f"\n[{self.name}] 开始按病人分组处理，共 {len(patient_groups)} 个病人")

        # 加载 table/ 文件夹中的 Excel 表格数据
        table_data = {}
        table_file = None
        try:
            from tools.table_reader import find_table_file, load_table_data
            table_file = find_table_file(dir_path)
            if table_file:
                table_data = load_table_data(table_file)
                if self.verbose:
                    print(f"[{self.name}] 加载表格数据: {os.path.basename(table_file)} "
                          f"({len(table_data)} 条记录)")
        except Exception as e:
            if self.verbose:
                print(f"[{self.name}] 加载表格数据失败（将跳过）: {e}")

        # 按病人处理
        all_results = {
            "success": True,
            "data_type": "directory",
            "root_path": dir_path,
            "processed_at": datetime.now().isoformat(),
            "statistics": {
                "total_patients": len(patient_groups),
                "total_files": analysis["statistics"]["image_files"],
                "processed_files": 0,
                "failed_files": 0
            },
            "table_file": table_file,
            "table_record_count": len(table_data),
            "patients": {}
        }

        for patient_id, patient_data in patient_groups.items():
            if self.verbose:
                print(f"\n[{self.name}] 处理病人 {patient_id} ({patient_data['file_count']} 个文件)")

            patient_result = await self._process_patient_files(
                patient_id,
                patient_data["files"],
                patient_data.get("categories", []),
                table_data=table_data
            )
            
            all_results["patients"][patient_id] = patient_result
            all_results["statistics"]["processed_files"] += patient_result.get("processed_count", 0)
            all_results["statistics"]["failed_files"] += patient_result.get("failed_count", 0)
        
        # 生成摘要
        description = self._format_directory_result(all_results)
        
        return Msg(
            self.name,
            content=description,
            role="assistant",
            metadata=all_results
        )
    
    async def _process_patient_files(
        self,
        patient_id: str,
        files: List[Dict],
        categories: List[str],
        table_data: Dict = None
    ) -> Dict[str, Any]:
        """
        处理单个病人的所有文件

        Args:
            patient_id: 病人ID
            files: 该病人的文件列表
            categories: 文件分类
            table_data: 全局表格数据（由 load_table_data 返回），用于补充填充

        Returns:
            病人处理结果
        """
        # 查找该病人在表格中的行数据
        patient_table = {}
        if table_data:
            try:
                from tools.table_reader import get_patient_table_data
                patient_table = get_patient_table_data(table_data, patient_id)
                if self.verbose and patient_table:
                    print(f"  - 找到表格数据: {len(patient_table)} 个字段")
                elif self.verbose:
                    print(f"  - 未在表格中找到病人 {patient_id} 的记录")
            except Exception as e:
                if self.verbose:
                    print(f"  - 表格数据查找失败: {e}")

        patient_result = {
            "patient_id": patient_id,
            "categories": categories,
            "file_count": len(files),
            "processed_count": 0,
            "failed_count": 0,
            "files": [],
            "table_data": patient_table,
            "merged_data": {
                "entities": [],
                "standardized_terms": []
            }
        }
        
        for file_info in files:
            file_path = file_info["path"]
            category = file_info.get("category", "unknown")
            source_type = file_info.get("source_type", "ocr")  # 获取来源类型
            
            if self.verbose:
                print(f"  - 处理: {os.path.basename(file_path)} ({category}, {source_type})")
            
            try:
                # 检测并处理单个文件
                data_type, _ = detect_data_type(file_path)
                
                if data_type == DataType.IMAGE:
                    # 根据 source_type 选择处理方式
                    if source_type == "figure":
                        # figure 类型：先版面分析和切割
                        result = await self._process_figure(file_path, category)
                    else:
                        # ocr 类型：普通 OCR 处理
                        result = await self._process_image(file_path)
                elif data_type == DataType.TEXT:
                    result = await self._process_text(file_path)
                else:
                    result = {"success": False, "error": f"不支持的文件类型: {data_type.value}"}
                
                file_result = {
                    "path": file_path,
                    "filename": os.path.basename(file_path),
                    "category": category,
                    "success": result.get("success", False),
                    "data_type": data_type.value
                }
                
                if result.get("success"):
                    patient_result["processed_count"] += 1
                    
                    # 保存OCR文本
                    if result.get("ocr_text"):
                        file_result["ocr_text"] = result["ocr_text"]
                    
                    # 保存预处理后的文本
                    if result.get("preprocessed_text"):
                        file_result["preprocessed_text"] = result["preprocessed_text"]
                    
                    # 从 extraction_result 中提取信息
                    extraction = result.get("extraction_result", {})
                    if isinstance(extraction, dict):
                        # 提取实体
                        entities = extraction.get("entities", [])
                        if entities:
                            file_result["extraction_result"] = entities
                            for entity in entities:
                                entity_copy = dict(entity)
                                entity_copy["source_file"] = os.path.basename(file_path)
                                entity_copy["folder_category"] = category
                                patient_result["merged_data"]["entities"].append(entity_copy)
                        
                        # 提取时间信息
                        temporal_info = extraction.get("temporal_info", [])
                        if temporal_info:
                            file_result["temporal_info"] = temporal_info
                        
                        # 提取剂量信息
                        quantity_info = extraction.get("quantity_info", [])
                        if quantity_info:
                            file_result["quantity_info"] = quantity_info
                        
                        # 提取关系信息
                        relations = extraction.get("relations", [])
                        if relations:
                            file_result["relations"] = relations
                    
                    # 从 standardized_result 中提取标准化结果
                    std_result = result.get("standardized_result", {})
                    if isinstance(std_result, dict):
                        std_entities = std_result.get("entities", [])
                        if std_entities:
                            file_result["standardized_result"] = std_entities
                            for entity in std_entities:
                                entity_copy = dict(entity)
                                entity_copy["source_file"] = os.path.basename(file_path)
                                entity_copy["folder_category"] = category
                                patient_result["merged_data"]["standardized_terms"].append(entity_copy)
                    
                    file_result["entities_count"] = len(extraction.get("entities", [])) if isinstance(extraction, dict) else 0
                else:
                    patient_result["failed_count"] += 1
                    file_result["error"] = result.get("error", "Unknown error")
                
                patient_result["files"].append(file_result)
                
            except Exception as e:
                patient_result["failed_count"] += 1
                patient_result["files"].append({
                    "path": file_path,
                    "filename": os.path.basename(file_path),
                    "category": category,
                    "success": False,
                    "error": str(e)
                })
                if self.verbose:
                    print(f"    ❌ 处理失败: {e}")
        
        return patient_result
    
    def _format_directory_result(self, result: Dict) -> str:
        """格式化目录处理结果"""
        lines = [
            f"📁 目录处理完成",
            f"",
            f"统计:",
            f"  - 总病人数: {result['statistics']['total_patients']}",
            f"  - 总文件数: {result['statistics']['total_files']}",
            f"  - 成功处理: {result['statistics']['processed_files']}",
            f"  - 处理失败: {result['statistics']['failed_files']}",
            f"",
            f"病人列表:"
        ]
        
        for pid, pdata in result.get("patients", {}).items():
            status = "✅" if pdata["failed_count"] == 0 else "⚠️"
            lines.append(
                f"  {status} 病人 {pid}: "
                f"{pdata['processed_count']}/{pdata['file_count']} 文件 "
                f"({', '.join(pdata.get('categories', []))})"
            )
        
        return "\n".join(lines)

    async def _process_csv(self, file_path: str) -> Dict[str, Any]:
        """
        处理CSV文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            处理结果
        """
        if self.verbose:
            print(f"[{self.name}] 使用CSV处理流程")
        
        # 如果有LLM，先让LLM分析CSV结构
        analysis_result = None
        if self.model and self.use_llm:
            analysis_result = await self._llm_analyze_csv_for_processing(file_path)
        
        # 调用CSV处理器
        if analysis_result:
            extraction_columns = analysis_result.get("extraction_columns", [])
            
            result = await self.csv_processor.process_file(
                file_path=file_path,
                standardization_columns=analysis_result.get("standardization_columns"),
                value_columns=analysis_result.get("value_columns"),
                unit_columns=analysis_result.get("unit_columns"),
                category_mapping=analysis_result.get("category_mapping"),
                extraction_columns=extraction_columns,
                max_rows=None  # 处理全部数据
            )
            result["llm_analysis"] = analysis_result
            
            # 根据是否有extraction_columns更新处理流程说明
            if extraction_columns:
                result["processing_stages"] = ["extraction", "standardization", "unit_normalization"]
            else:
                result["processing_stages"] = ["standardization", "unit_normalization"]
        else:
            # 使用自动分析
            result = await self.csv_processor.process_file(
                file_path=file_path,
                max_rows=None
            )
            # 检查是否有extraction_columns
            if result.get("statistics", {}).get("extraction_columns"):
                result["processing_stages"] = ["extraction", "standardization", "unit_normalization"]
            else:
                result["processing_stages"] = ["standardization", "unit_normalization"]
        
        result["data_type"] = "csv"
        
        return result
    
    async def _llm_analyze_csv_for_processing(self, file_path: str) -> Optional[Dict]:
        """使用LLM分析CSV文件结构"""
        try:
            csv_info = get_csv_info_for_agent(file_path)
            
            prompt = f"""你是一个医学数据分析专家。请分析以下CSV文件，判断哪些列需要进行不同类型的处理。

{csv_info}

请返回JSON格式的分析结果：
{{
    "standardization_columns": ["需要术语标准化的列名（已结构化的短文本列，如疾病名、药品名、检验项目名等）"],
    "value_columns": ["需要量纲统一的数值列名"],
    "unit_columns": {{"数值列名": "对应的单位列名"}},
    "category_mapping": {{"列名": "类别"}},
    "extraction_columns": ["需要进行信息抽取的列名（包含自由文本/长文本的列，如报告正文、出院记录、病历描述等）"],
    "reasoning": "分析理由"
}}

判断规则：
1. **standardization_columns**: 选择已结构化的短文本列（如test_name, org_name, ab_name等），这些列的值通常较短且已是术语
2. **extraction_columns**: 选择包含自由文本的列（如text, note, report, findings等），这些列通常包含完整的医学报告或描述，需要先进行信息抽取
3. **value_columns**: 数值列，如果有对应的单位列需要进行量纲统一
4. ID列、日期列、时间列不需要任何处理
5. 类别可以是：Disease, Drug, Test, Symptom, Anatomy, Finding

特别注意：
- 如果列名包含 text, note, report, description, findings, impression, narrative 等词，且内容较长，应放入 extraction_columns
- 短的术语列（如 test_name, drug_name）应放入 standardization_columns

只返回JSON。"""

            import openai
            config = get_api_config()
            client = openai.AsyncOpenAI(
                api_key=config['api_key'],
                base_url=config['api_base']
            )
            
            response = await client.chat.completions.create(
                model=config['model_name'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            
            # 提取JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
            
        except Exception as e:
            if self.verbose:
                print(f"[{self.name}] LLM分析CSV失败: {e}")
            return None
    
    async def _process_image(self, file_path: str) -> Dict[str, Any]:
        """
        处理图片文件
        
        Args:
            file_path: 图片文件路径
            
        Returns:
            处理结果
        """
        if self.verbose:
            print(f"[{self.name}] 使用图片处理流程: OCR -> 预处理 -> 信息抽取 -> 标准化")
        
        if not self.med_agent or not self.std_agent:
            return {
                "success": False,
                "error": "图片处理Agent未初始化，请检查LLM配置",
                "data_type": "image"
            }
        
        result = {
            "success": False,
            "data_type": "image",
            "file_path": file_path,
            "processing_stages": ["ocr", "preprocessing", "extraction", "standardization", "unit_normalization"],
            "ocr_text": None,
            "preprocessed_text": None,
            "extraction_result": None,
            "standardized_result": None
        }
        
        try:
            # Step 1: 先单独进行OCR获取原始文本
            if self.verbose:
                print(f"[{self.name}] Step 1: OCR识别...")
            
            try:
                from tools.tools_ocr import extract_text_from_image
                from tools.tools_preprocess import clean_medical_text
                
                # 获取OCR原始文本
                ocr_result = extract_text_from_image(file_path)
                if ocr_result.content and len(ocr_result.content) > 0:
                    ocr_text = ocr_result.content[0].get("text", "")
                    if ocr_text and not ocr_text.startswith("Error:"):
                        result["ocr_text"] = ocr_text
                        
                        # 清洗文本
                        cleaned_result = clean_medical_text(ocr_text)
                        if cleaned_result.content and len(cleaned_result.content) > 0:
                            result["preprocessed_text"] = cleaned_result.content[0].get("text", "")
                        
                        if self.verbose:
                            print(f"[{self.name}] OCR完成，识别到 {len(ocr_text)} 字符")
            except ImportError as e:
                if self.verbose:
                    print(f"[{self.name}] OCR模块导入失败: {e}，使用Agent处理")
            except Exception as e:
                if self.verbose:
                    print(f"[{self.name}] OCR处理异常: {e}，使用Agent处理")
            
            # Step 2: 使用医学Agent进行信息抽取
            if self.verbose:
                print(f"[{self.name}] Step 2: 信息抽取...")
            
            # 如果已有预处理文本，使用文本作为输入；否则使用文件路径
            if result["preprocessed_text"]:
                msg = Msg(name="User", content=result["preprocessed_text"], role="user")
            else:
                msg = Msg(name="User", content=file_path, role="user")
            
            # 信息提取
            res = await self.med_agent(msg, structured_model=self.extraction_schema)
            
            # 如果之前OCR失败，尝试从Agent响应中获取文本
            if not result["ocr_text"] and hasattr(res, 'content') and res.content:
                content = res.content
                if isinstance(content, str) and len(content) > 50:
                    result["ocr_text"] = content
            
            if res.metadata:
                result["extraction_result"] = res.metadata
                
                # 如果之前没有预处理文本，从entities的original_text构建
                entities = res.metadata.get("entities", [])
                if not result["preprocessed_text"] and entities:
                    original_texts = []
                    for entity in entities:
                        orig_text = entity.get("original_text", "")
                        if orig_text and orig_text not in original_texts:
                            original_texts.append(orig_text)
                    if original_texts:
                        result["preprocessed_text"] = "; ".join(original_texts)
                
                if self.verbose:
                    entities_count = len(entities)
                    print(f"[{self.name}] 提取到 {entities_count} 个实体")
                    print(f"[{self.name}] Step 3: 术语标准化和量纲统一...")
                
                # 标准化
                std_res = await self.std_agent(res)
                
                if std_res.metadata:
                    result["standardized_result"] = std_res.metadata
                    result["success"] = True
                    
                    if self.verbose:
                        stats = std_res.metadata.get("_standardization_stats", {})
                        print(f"[{self.name}] 标准化完成: {stats.get('standardized', 0)}/{stats.get('total', 0)} 实体已标准化, {stats.get('unit_normalized', 0)} 个单位已转换")
                else:
                    result["standardized_result"] = {"content": std_res.content}
                    result["success"] = True
            else:
                # 没有结构化结果，但可能有OCR文本
                result["extraction_result"] = {"content": res.content}
                if not result["ocr_text"] and res.content and len(res.content) > 50:
                    result["ocr_text"] = res.content
                result["success"] = True
                if self.verbose:
                    print(f"[{self.name}] 警告: 未获得结构化提取结果，跳过标准化")
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    async def _process_figure(self, file_path: str, category: str = None) -> Dict[str, Any]:
        """
        处理 figure 类型的图像文件（直接 OCR，不做版面分析）
        """
        if self.verbose:
            print(f"[{self.name}] Figure 处理流程: OCR -> 信息抽取 -> 标准化")

        result = await self._process_image(file_path)
        result["data_type"] = "figure"
        result["category"] = category
        return result
    
    async def _process_text(self, input_path: str) -> Dict[str, Any]:
        """
        处理文本数据
        
        Args:
            input_path: 文本文件路径或文本内容
            
        Returns:
            处理结果
        """
        if self.verbose:
            print(f"[{self.name}] 使用文本处理流程: 预处理 -> 信息抽取 -> 标准化")
        
        if not self.med_agent or not self.std_agent:
            return {
                "success": False,
                "error": "文本处理Agent未初始化，请检查LLM配置",
                "data_type": "text"
            }
        
        result = {
            "success": False,
            "data_type": "text",
            "file_path": input_path if os.path.exists(input_path) else None,
            "processing_stages": ["preprocessing", "extraction", "standardization", "unit_normalization"],
            "preprocessed_text": None,
            "extraction_result": None,
            "standardized_result": None
        }
        
        try:
            msg = Msg(name="User", content=input_path, role="user")
            
            # 信息提取
            res = await self.med_agent(msg, structured_model=self.extraction_schema)
            
            if res.metadata:
                result["extraction_result"] = res.metadata
                
                # 标准化
                std_res = await self.std_agent(res)
                
                if std_res.metadata:
                    result["standardized_result"] = std_res.metadata
                    result["success"] = True
                else:
                    result["standardized_result"] = {"content": std_res.content}
                    result["success"] = True
            else:
                result["extraction_result"] = {"content": res.content}
                result["success"] = True
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _format_result(self, result: Dict) -> str:
        """格式化处理结果"""
        lines = [
            f"=== 处理结果 ===",
            f"数据类型: {result.get('data_type', '未知')}",
            f"处理状态: {'成功' if result.get('success') else '失败'}",
        ]
        
        if result.get("error"):
            lines.append(f"错误信息: {result['error']}")
        
        if result.get("processing_stages"):
            lines.append(f"处理流程: {' -> '.join(result['processing_stages'])}")
        
        # CSV处理统计
        if result.get("statistics"):
            stats = result["statistics"]
            lines.extend([
                f"",
                f"--- 处理统计 ---",
                f"总行数: {stats.get('total_rows', 'N/A')}",
                f"术语标准化: {stats.get('terms_standardized', 0)} 成功, {stats.get('terms_not_found', 0)} 未找到",
                f"量纲统一: {stats.get('values_normalized', 0)} 已转换, {stats.get('values_unchanged', 0)} 保持不变",
            ])
        
        # 图片/文本处理的标准化结果
        if result.get("standardized_result"):
            std_result = result["standardized_result"]
            lines.append(f"")
            lines.append(f"--- 标准化结果 ---")
            
            # 显示标准化统计
            if "_standardization_stats" in std_result:
                stats = std_result["_standardization_stats"]
                lines.append(f"术语标准化: {stats.get('standardized', 0)}/{stats.get('total', 0)} 已标准化")
                lines.append(f"量纲统一: {stats.get('unit_normalized', 0)} 个单位已转换")
            
            # 显示已标准化的实体
            if "entities" in std_result:
                standardized_entities = [
                    e for e in std_result["entities"] 
                    if e.get("standard_code") or e.get("normalized_value")
                ]
                if standardized_entities:
                    lines.append(f"")
                    lines.append(f"已标准化的实体 ({len(standardized_entities)}个):")
                    for entity in standardized_entities[:10]:  # 最多显示10个
                        name = entity.get("name", "")
                        category = entity.get("category", "")
                        std_code = entity.get("standard_code", "")
                        std_system = entity.get("standard_system", "")
                        std_name = entity.get("standard_name", "")
                        
                        # 量纲统一信息
                        orig_val = entity.get("value", "")
                        orig_unit = entity.get("unit", "")
                        norm_val = entity.get("normalized_value", "")
                        norm_unit = entity.get("normalized_unit", "")
                        
                        line_parts = [f"  • {name} ({category})"]
                        if std_code:
                            line_parts.append(f" → [{std_system}] {std_code}")
                            if std_name:
                                line_parts.append(f" ({std_name})")
                        if norm_val and norm_unit and str(norm_val) != str(orig_val):
                            line_parts.append(f" | {orig_val} {orig_unit} → {norm_val} {norm_unit}")
                        
                        lines.append("".join(line_parts))
                    
                    if len(standardized_entities) > 10:
                        lines.append(f"  ... 还有 {len(standardized_entities) - 10} 个")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict:
        """获取处理统计"""
        return self.processing_stats.copy()


async def process_medical_data(
    input_path: str,
    use_llm: bool = True,
    use_umls: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    处理医学数据的便捷函数
    
    自动识别数据类型并选择合适的处理流程。
    
    Args:
        input_path: 输入文件路径或文本内容
        use_llm: 是否使用LLM
        use_umls: 是否使用UMLS
        verbose: 是否详细输出
        
    Returns:
        处理结果
    """
    agent = UnifiedProcessingAgent(
        use_llm=use_llm,
        use_umls=use_umls,
        verbose=verbose
    )
    
    msg = Msg(name="User", content=input_path, role="user")
    result = await agent.reply(msg)
    
    return result.metadata if result.metadata else {"content": result.content}
