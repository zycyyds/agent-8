# -*- coding: utf-8 -*-
"""
数据类型检测Agent

使用LLM自动识别输入数据的类型和特征，
决定使用哪种处理流程。
"""
import os
import sys
import json
from typing import Dict, Optional, Any, List

# 设置导入路径
module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

parent_dir = os.path.dirname(module_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 添加agentscope src路径
src_path = os.path.abspath(os.path.join(parent_dir, "../../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.tool import Toolkit

from tools.data_type_detector import detect_input_type
from tools.csv_reader import get_csv_info_for_agent, read_csv_sample, analyze_columns_for_standardization
from config.settings import DataType, ProcessingStage, PROCESSING_PIPELINE


class DataTypeDetectorAgent(AgentBase):
    """
    数据类型检测Agent
    
    功能：
    1. 自动检测输入数据的类型（图片/文本/CSV）
    2. 对于CSV文件，分析列结构并判断哪些列需要处理
    3. 返回处理建议，供后续Agent使用
    """
    
    def __init__(
        self,
        name: str = "DataTypeDetector",
        model: Optional[Any] = None,
        use_llm_for_csv_analysis: bool = True,
        verbose: bool = False
    ):
        """
        初始化Agent
        
        Args:
            name: Agent名称
            model: LLM模型（用于CSV分析时进行智能判断）
            use_llm_for_csv_analysis: 是否使用LLM分析CSV结构
            verbose: 是否打印详细信息
        """
        super().__init__()
        self.name = name
        self.model = model
        self.use_llm_for_csv_analysis = use_llm_for_csv_analysis
        self.verbose = verbose
        
        # 注册工具
        self.toolkit = Toolkit()
        self.toolkit.register_tool_function(
            detect_input_type,
            func_description="检测输入数据的类型（图片/文本/CSV）"
        )
        self.toolkit.register_tool_function(
            get_csv_info_for_agent,
            func_description="获取CSV文件的详细分析报告，包括列名、样本数据和标准化建议"
        )
    
    async def reply(self, msg: Msg | dict | None = None) -> Msg:
        """
        处理输入消息，返回数据类型检测结果
        
        Args:
            msg: 输入消息，包含文件路径或数据
            
        Returns:
            包含检测结果的消息
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
            print(f"[{self.name}] 检测输入: {input_path}")
        
        # Step 1: 基础类型检测
        detection_result = detect_input_type(input_path)
        data_type = DataType(detection_result["type"])
        
        if self.verbose:
            print(f"[{self.name}] 检测到类型: {data_type.value}")
        
        # Step 2: 对于CSV文件，进行详细分析
        csv_analysis = None
        processing_recommendation = None
        
        if data_type == DataType.CSV:
            csv_analysis = await self._analyze_csv(input_path)
            processing_recommendation = self._generate_csv_recommendation(csv_analysis)
        else:
            processing_recommendation = self._generate_default_recommendation(data_type)
        
        # 构建结果
        result = {
            "data_type": data_type.value,
            "detection_details": detection_result,
            "processing_stages": detection_result.get("processing_stages", []),
            "recommendation": processing_recommendation,
        }
        
        if csv_analysis:
            result["csv_analysis"] = csv_analysis
        
        # 生成人类可读的描述
        description = self._format_result(result)
        
        return Msg(
            self.name,
            content=description,
            role="assistant",
            metadata=result
        )
    
    async def _analyze_csv(self, file_path: str) -> Dict[str, Any]:
        """
        分析CSV文件结构
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            分析结果字典
        """
        analysis = {
            "file_path": file_path,
            "columns": [],
            "sample_data": [],
            "standardization_candidates": [],
            "unit_normalization_candidates": [],
            "unit_columns": [],
            "llm_analysis": None
        }
        
        # 读取样本
        sample_result = read_csv_sample(file_path, sample_rows=5)
        if not sample_result["success"]:
            analysis["error"] = sample_result["error"]
            return analysis
        
        analysis["columns"] = sample_result["columns"]
        analysis["sample_data"] = sample_result["sample_data"]
        analysis["total_rows"] = sample_result["total_rows"]
        
        # 基于规则的分析
        rule_analysis = analyze_columns_for_standardization(
            sample_result["columns"],
            sample_result["sample_data"]
        )
        
        analysis["standardization_candidates"] = rule_analysis["standardization_candidates"]
        analysis["unit_normalization_candidates"] = rule_analysis["unit_normalization_candidates"]
        analysis["unit_columns"] = rule_analysis["unit_columns"]
        
        # 如果启用LLM分析
        if self.use_llm_for_csv_analysis and self.model:
            llm_analysis = await self._llm_analyze_csv(
                sample_result["columns"],
                sample_result["sample_data"]
            )
            analysis["llm_analysis"] = llm_analysis
            
            # 合并LLM的建议
            if llm_analysis:
                self._merge_llm_suggestions(analysis, llm_analysis)
        
        return analysis
    
    async def _llm_analyze_csv(
        self,
        columns: List[str],
        sample_data: List[List[str]]
    ) -> Optional[Dict]:
        """
        使用LLM分析CSV结构
        
        Args:
            columns: 列名列表
            sample_data: 样本数据
            
        Returns:
            LLM分析结果
        """
        if not self.model:
            return None
        
        # 构建提示
        sample_str = "\n".join([
            ", ".join(f"{col}: {val}" for col, val in zip(columns, row))
            for row in sample_data[:3]
        ])
        
        prompt = f"""你是一个医学数据分析专家。请分析以下CSV文件的结构，判断哪些列需要进行标准化处理。

列名：
{json.dumps(columns, ensure_ascii=False)}

样本数据（前3行）：
{sample_str}

请分析并返回JSON格式的结果：
{{
    "standardization_columns": ["需要术语标准化的列名列表，这些列包含医学术语如疾病、药品、检验项目等"],
    "value_columns": ["需要量纲统一的数值列名列表"],
    "unit_columns": {{"数值列名": "对应的单位列名"}},
    "category_mapping": {{"列名": "类别"}},  // 类别可以是 Disease, Drug, Test, Symptom 等
    "reasoning": "分析理由"
}}

注意：
1. 只选择确实包含医学术语的列进行标准化
2. 对于已经是ID或编码的列（如subject_id），不需要标准化
3. 如果某列包含浓度、数量等数值，且有对应的单位，则需要量纲统一

只返回JSON，不要其他文字。"""

        try:
            import openai
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
            api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            
            client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)
            response = await client.chat.completions.create(
                model="gpt-4.1-mini",
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
                print(f"[{self.name}] LLM分析失败: {e}")
            return None
    
    def _merge_llm_suggestions(self, analysis: Dict, llm_result: Dict) -> None:
        """将LLM的建议合并到分析结果中"""
        # 添加LLM建议的标准化列（如果规则没有检测到）
        existing_std_cols = {item["column"] for item in analysis["standardization_candidates"]}
        for col in llm_result.get("standardization_columns", []):
            if col not in existing_std_cols and col in analysis["columns"]:
                analysis["standardization_candidates"].append({
                    "column": col,
                    "index": analysis["columns"].index(col),
                    "source": "llm"
                })
        
        # 添加LLM建议的量纲列
        existing_val_cols = {item["column"] for item in analysis["unit_normalization_candidates"]}
        for col in llm_result.get("value_columns", []):
            if col not in existing_val_cols and col in analysis["columns"]:
                analysis["unit_normalization_candidates"].append({
                    "column": col,
                    "index": analysis["columns"].index(col),
                    "source": "llm"
                })
        
        # 保存类别映射
        if "category_mapping" in llm_result:
            analysis["category_mapping"] = llm_result["category_mapping"]
        
        # 保存单位列映射
        if "unit_columns" in llm_result:
            analysis["unit_column_mapping"] = llm_result["unit_columns"]
    
    def _generate_csv_recommendation(self, analysis: Dict) -> Dict:
        """为CSV数据生成处理建议"""
        return {
            "processing_type": "csv",
            "skip_preprocessing": True,
            "skip_extraction": True,
            "standardization": {
                "enabled": len(analysis.get("standardization_candidates", [])) > 0,
                "columns": [item["column"] for item in analysis.get("standardization_candidates", [])],
                "category_mapping": analysis.get("category_mapping", {})
            },
            "unit_normalization": {
                "enabled": len(analysis.get("unit_normalization_candidates", [])) > 0,
                "value_columns": [item["column"] for item in analysis.get("unit_normalization_candidates", [])],
                "unit_column_mapping": analysis.get("unit_column_mapping", {})
            },
            "description": self._generate_recommendation_description(analysis)
        }
    
    def _generate_default_recommendation(self, data_type: DataType) -> Dict:
        """为非CSV数据生成默认处理建议"""
        stages = [s.value for s in PROCESSING_PIPELINE.get(data_type, [])]
        
        return {
            "processing_type": data_type.value,
            "skip_preprocessing": False,
            "skip_extraction": False,
            "processing_stages": stages,
            "description": f"将按照{data_type.value}类型的标准流程处理: {' -> '.join(stages)}"
        }
    
    def _generate_recommendation_description(self, analysis: Dict) -> str:
        """生成处理建议的描述"""
        lines = ["CSV数据将跳过预处理和信息抽取，直接进行标准化和量纲统一。"]
        
        std_cols = [item["column"] for item in analysis.get("standardization_candidates", [])]
        if std_cols:
            lines.append(f"需要术语标准化的列: {', '.join(std_cols)}")
        
        val_cols = [item["column"] for item in analysis.get("unit_normalization_candidates", [])]
        if val_cols:
            lines.append(f"需要量纲统一的列: {', '.join(val_cols)}")
        
        return " ".join(lines)
    
    def _format_result(self, result: Dict) -> str:
        """格式化检测结果为可读字符串"""
        lines = [
            f"=== 数据类型检测结果 ===",
            f"检测到类型: {result['data_type']}",
            f"处理阶段: {' -> '.join(result['processing_stages'])}",
            "",
        ]
        
        if result.get("csv_analysis"):
            csv_info = result["csv_analysis"]
            lines.extend([
                f"--- CSV文件信息 ---",
                f"总行数: {csv_info.get('total_rows', '未知')}",
                f"列数: {len(csv_info.get('columns', []))}",
                f"列名: {', '.join(csv_info.get('columns', [])[:10])}{'...' if len(csv_info.get('columns', [])) > 10 else ''}",
                "",
            ])
        
        rec = result.get("recommendation", {})
        if rec.get("description"):
            lines.append(f"处理建议: {rec['description']}")
        
        return "\n".join(lines)
