"""项目主流程：读取数据、生成任务规格、筛列并导出结果。"""
import os
import json
import sys
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import re
import glob

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from .agents.task_planner import TaskPlannerAgent
    from .agents.column_selector import ColumnSelectorAgent
    from .components.schema_profiler import SchemaProfiler
    from .components.hard_gates import HardGates
    from .config_manager import get_default_model_name
except ImportError:
    from medical_column_selector.agents.task_planner import TaskPlannerAgent
    from medical_column_selector.agents.column_selector import ColumnSelectorAgent
    from medical_column_selector.components.schema_profiler import SchemaProfiler
    from medical_column_selector.components.hard_gates import HardGates
    from medical_column_selector.config_manager import get_default_model_name


class TaskDrivenColumnSelector:
    """任务驱动的医疗表格筛列器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化主流程对象，并准备两个 agent。"""
        # AgentScope 只需要初始化一次最小配置即可，不在这里重复解析模型配置。
        try:
            import agentscope

            agentscope.init(
                logging_level="INFO",
                project="medical_column_selector",
                name="medical_column_selector_main"
            )
        except Exception as e:
            print(f"Warning: Could not initialize agentscope properly: {e}")

        self.config = config or {}
        self.default_config = {
            "privacy_mode": "research",
            "keep_free_text": False,
            "max_rows_profile": 1000,
            "max_final_columns": 100,
            "sampling_method": "random",
            "random_seed": 42,
            "input_sheet_name": None,
            "model_name": self._get_default_model_name(),
            "require_llm": True,
            "selector_input_mode": "full_schema",
        }

        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        model_name = self.config.get("model_name", self._get_default_model_name())
        allow_fallback = not self.config.get("require_llm", True)
        self.task_planner_agent = TaskPlannerAgent(model_name=model_name, allow_fallback=allow_fallback)
        self.column_selector_agent = ColumnSelectorAgent(model_name=model_name, allow_fallback=allow_fallback)

        if self.config.get("require_llm", True):
            self.task_planner_agent.ensure_llm_ready()
            self.column_selector_agent.ensure_llm_ready()

    def _get_default_model_name(self) -> str:
        """从配置文件里拿默认模型名。"""
        return get_default_model_name()

    def _extract_task_keywords(self, task_text: str) -> List[str]:
        """把任务文本压成一组简短关键词，用于候选列召回。"""
        text = (task_text or "").lower()
        tokens = re.findall(r"[a-z][a-z0-9_]{1,}|[\u4e00-\u9fff]{2,}", text)
        stopwords = {
            "identify", "analyze", "analysis", "study", "patient", "patients",
            "with", "for", "from", "into", "using", "based", "task", "medical",
            "data", "and", "the", "of", "to", "in", "on", "a", "an",
            "进行", "分析", "患者", "数据", "研究", "筛选", "识别", "用于", "相关",
        }

        deduped = []
        for token in tokens:
            if token in stopwords:
                continue
            if token not in deduped:
                deduped.append(token)
            if len(deduped) >= 25:
                break
        return deduped

    def _read_input_dataframe(self, input_path: str) -> pd.DataFrame:
        """统一读取 CSV / Excel，避免分析和导出逻辑不一致。"""
        return SchemaProfiler()._read_tabular_file(
            input_path,
            sheet_name=self.config.get("input_sheet_name"),
        )

    def _structured_table_modalities(self, col_name: str) -> List[str]:
        """给中文结构化表头做稳定 modality 映射。"""
        col_lower = (col_name or "").lower()
        modalities: List[str] = []

        if col_lower.startswith("table_一般资料"):
            modalities.append("Demographics")
            if any(token in col_lower for token in ["出生年月", "年龄", "日期"]):
                modalities.append("Time")

        if col_lower.startswith("table_症状学"):
            modalities.append("Diagnosis")

        if col_lower.startswith("table_床旁查体"):
            modalities.append("Procedures")

        if col_lower.startswith("table_初步诊断") or col_lower.startswith("table_第一次复诊"):
            modalities.append("Diagnosis")

        if col_lower.startswith("table_辅助检查"):
            modalities.append("Labs")
            if any(token in col_lower for token in ["查体", "hit试验", "dix-hallpike", "roll-test", "位置试验"]):
                modalities.append("Procedures")
            if any(token in col_lower for token in ["诊断日期", "持续时间"]):
                modalities.append("Time")

        deduped: List[str] = []
        for modality in modalities:
            if modality not in deduped:
                deduped.append(modality)
        return deduped

    def _is_structured_clinical_table_column(self, col_name: str) -> bool:
        """判断是否属于需要稳定召回的中文结构化表字段。"""
        col_lower = (col_name or "").lower()
        return col_lower.startswith((
            "table_一般资料",
            "table_症状学",
            "table_辅助检查",
            "table_初步诊断",
            "table_第一次复诊",
            "table_床旁查体",
        ))

    def _is_task_relevant_structured_column(
        self,
        col_name: str,
        task_keywords: List[str],
        need_modalities: set[str],
    ) -> bool:
        """对中文结构化字段做补充召回，避免重要列被漏掉。"""
        structured_modalities = set(self._structured_table_modalities(col_name))
        if structured_modalities & need_modalities:
            return True

        col_lower = (col_name or "").lower()
        vestibular_tokens = [
            "眩晕", "头晕", "前庭", "眼震", "眼动", "平衡", "步态",
            "hit试验", "dix-hallpike", "roll-test", "冷热试验",
            "耳鸣", "复视", "姿势", "踏步试验"
        ]
        if any(token in col_lower for token in vestibular_tokens):
            return True

        return any(keyword in col_lower for keyword in task_keywords)

    def _build_llm_input_profile(
        self,
        schema_profile: Dict[str, Any],
        task_spec: Dict[str, Any],
        task_text: str
    ) -> Dict[str, Any]:
        """给列筛选阶段准备 schema，默认把完整候选集交给 LLM。"""
        all_columns = schema_profile.get("columns", {})
        privacy_mode = self.config.get("privacy_mode", "research")
        task_keywords = self._extract_task_keywords(task_text)
        need_modalities = set(task_spec.get("need_modalities", []))
        selector_input_mode = self.config.get("selector_input_mode", "full_schema")

        prefiltered_cols: Dict[str, Dict[str, Any]] = {}
        dropped_constant_or_missing: List[str] = []
        dropped_phi_in_strict: List[str] = []

        for col_name, col_profile in all_columns.items():
            is_all_missing = col_profile.get("missing_rate", 0) == 1.0
            is_constant = col_profile.get("is_constant", False)
            is_identifier = col_profile.get("inferred_modality") == "Identifiers/Keys"

            if is_all_missing:
                dropped_constant_or_missing.append(col_name)
                continue

            if is_constant and not is_identifier:
                dropped_constant_or_missing.append(col_name)
                continue

            if privacy_mode == "strict" and col_profile.get("is_potential_phi", False):
                dropped_phi_in_strict.append(col_name)
                continue

            prefiltered_cols[col_name] = col_profile

        if not prefiltered_cols:
            prefiltered_cols = dict(all_columns)

        candidate_cols: List[str]
        if selector_input_mode == "heuristic_prefilter":
            candidate_cols = []
            for col_name, col_profile in prefiltered_cols.items():
                col_lower = col_name.lower()
                modality = col_profile.get("inferred_modality")

                modality_hit = modality in need_modalities
                keyword_hit = any(keyword in col_lower for keyword in task_keywords)
                baseline_hit = modality in {"Identifiers/Keys", "Time"}
                structured_hit = self._is_structured_clinical_table_column(col_name) and self._is_task_relevant_structured_column(
                    col_name,
                    task_keywords=task_keywords,
                    need_modalities=need_modalities,
                )

                if modality_hit or keyword_hit or baseline_hit or structured_hit:
                    candidate_cols.append(col_name)

            min_candidates = min(12, len(prefiltered_cols))
            if len(candidate_cols) < min_candidates:
                for col_name, col_profile in prefiltered_cols.items():
                    if col_name in candidate_cols:
                        continue
                    if col_profile.get("inferred_modality") in {
                        "Demographics", "Encounter", "Outcomes", "Vitals",
                        "Diagnosis", "Labs", "Medication", "Procedures"
                    }:
                        candidate_cols.append(col_name)
                    if len(candidate_cols) >= min_candidates:
                        break

            if not candidate_cols:
                candidate_cols = list(prefiltered_cols.keys())
        else:
            candidate_cols = list(prefiltered_cols.keys())

        candidate_set = set(candidate_cols)
        candidate_schema_columns = {
            col_name: col_profile
            for col_name, col_profile in prefiltered_cols.items()
            if col_name in candidate_set
        }

        llm_input_profile = {
            "dataset_info": {
                **schema_profile.get("dataset_info", {}),
                "total_columns_before_prefilter": len(all_columns),
                "columns_after_prefilter": len(prefiltered_cols),
                "columns_passed_to_selector": len(candidate_schema_columns),
                "dropped_constant_or_all_missing_count": len(dropped_constant_or_missing),
                "dropped_phi_in_strict_count": len(dropped_phi_in_strict),
                "task_keywords_used": task_keywords,
                "selector_input_mode": selector_input_mode,
            },
            "columns": candidate_schema_columns,
        }
        return llm_input_profile

    def run(self, input_csv_path: str, task_text: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        """执行完整筛列流程，并导出 CSV 和报告。"""
        print(f"Starting task-driven column selection for: {task_text}")

        if output_dir is None:
            output_dir = os.path.dirname(input_csv_path) or "."

        # 1. 先分析原始表结构。
        print("Step 1: Running Schema Profiler...")
        profiler = SchemaProfiler(
            max_rows=self.config.get("max_rows_profile", 1000),
            sampling_method=self.config.get("sampling_method", "random"),
            random_seed=self.config.get("random_seed", 42)
        )
        schema_profile = profiler.analyze_file(
            input_csv_path,
            sheet_name=self.config.get("input_sheet_name"),
        )

        # 2. 把自然语言任务转成结构化任务规格。
        print("Step 2: Running Task Planner Agent...")
        task_spec = self.task_planner_agent.generate_task_spec(task_text)
        task_spec["privacy_mode"] = self.config.get("privacy_mode", "research")

        # 3. 先缩小候选列范围，避免把整张表都送入后续筛选。
        print("Step 3: Building candidate columns for selector...")
        llm_input_profile = self._build_llm_input_profile(
            schema_profile=schema_profile,
            task_spec=task_spec,
            task_text=task_text
        )
        candidate_count = llm_input_profile["dataset_info"]["columns_passed_to_selector"]
        print(f"  - Candidate columns passed to selector: {candidate_count}")

        # 4. 让列筛选器输出 must/useful/maybe/drop。
        print("Step 4: Running Column Selector Agent...")
        selection_result = self.column_selector_agent.select_columns(
            task_spec=task_spec,
            schema_profile=llm_input_profile
        )

        # 5. 再套一层硬规则，处理 PHI、自由文本、常量列等。
        print("Step 5: Applying Hard Gates...")
        hard_gates = HardGates(
            privacy_mode=self.config.get("privacy_mode", "research"),
            keep_free_text=self.config.get("keep_free_text", False),
            max_final_columns=self.config.get("max_final_columns")
        )
        final_selection = hard_gates.apply_gates(
            selection_result=selection_result,
            schema_profile=schema_profile
        )

        # 6. 导出筛选后的数据和完整审计报告。
        print("Step 6: Exporting results...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_csv_path))[0]

        filtered_csv_path = os.path.join(output_dir, f"{base_name}_filtered_{timestamp}.csv")
        selection_report_path = os.path.join(output_dir, f"{base_name}_selection_report_{timestamp}.json")
        final_columns = final_selection["final_columns"]

        selection_report = {
            "timestamp": datetime.now().isoformat(),
            "task_text": task_text,
            "task_spec": task_spec,
            "column_counts": {
                "original_column_count": len(schema_profile.get("columns", {})),
                "prefiltered_column_count": llm_input_profile["dataset_info"].get("columns_after_prefilter", 0),
                "selector_candidate_column_count": llm_input_profile["dataset_info"].get("columns_passed_to_selector", 0),
                "final_column_count": len(final_columns),
            },
            "agent_execution": {
                "task_planner_mode": self.task_planner_agent.last_call_mode,
                "task_planner_error": self.task_planner_agent.last_call_error,
                "task_planner_raw_response": self.task_planner_agent.last_raw_response,
                "column_selector_mode": self.column_selector_agent.last_call_mode,
                "column_selector_error": self.column_selector_agent.last_call_error,
                "column_selector_raw_response": self.column_selector_agent.last_raw_response,
                "column_selector_normalization_info": self.column_selector_agent.last_normalization_info,
            },
            "schema_profile": schema_profile,
            "llm_input_profile": llm_input_profile,
            "agent_decisions": selection_result,
            "hard_gate_overrides": final_selection.get("overrides", {}),
            "final_columns": final_columns,
            "warnings": final_selection.get("warnings", []),
            "config_used": self.config
        }

        with open(selection_report_path, 'w', encoding='utf-8') as f:
            json.dump(selection_report, f, indent=2, ensure_ascii=False)

        if not final_columns:
            raise RuntimeError(
                "Column selector produced no final columns. "
                f"See selection report for diagnostics: {selection_report_path}"
            )

        df = self._read_input_dataframe(input_csv_path)
        filtered_df = df[final_columns]
        filtered_df.to_csv(filtered_csv_path, index=False)

        column_counts = selection_report["column_counts"]
        print(
            "  - Column counts: "
            f"{column_counts['original_column_count']} -> "
            f"{column_counts['prefiltered_column_count']} -> "
            f"{column_counts['selector_candidate_column_count']} -> "
            f"{column_counts['final_column_count']}"
        )
        print(f"Results saved:\n  - Filtered CSV: {filtered_csv_path}\n  - Selection Report: {selection_report_path}")

        return {
            "filtered_csv_path": filtered_csv_path,
            "selection_report_path": selection_report_path
        }

    def run_on_directory_csv(self, directory_path: str, csv_pattern: str, task_text: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Execute the workflow on a specific CSV file in a directory

        Args:
            directory_path: Path to the directory containing CSV files
            csv_pattern: Pattern to match the CSV file (e.g., "input_cleaned_38.csv")
            task_text: Medical research/diagnosis/task text description
            output_dir: Output directory for results (optional)

        Returns:
            Dict with paths to filtered CSV and selection report
        """
        # Find the specific CSV file
        target_path = os.path.join(directory_path, csv_pattern)

        if not os.path.exists(target_path):
            # If exact match not found, try glob pattern matching
            matching_files = glob.glob(os.path.join(directory_path, csv_pattern))

            if not matching_files:
                raise FileNotFoundError(f"No CSV file matching pattern '{csv_pattern}' found in directory: {directory_path}")

            # Use the first matching file
            target_path = matching_files[0]
            print(f"Found matching file: {target_path}")

        return self.run(target_path, task_text, output_dir)


def run_example():
    """Example usage of the TaskDrivenColumnSelector"""

    # Create sample CSV data for demonstration
    sample_data = {
        "patient_id": ["P001", "P002", "P003", "P004", "P005"],
        "first_name": ["John", "Jane", "Bob", "Alice", "Charlie"],
        "last_name": ["Doe", "Smith", "Johnson", "Brown", "Wilson"],
        "dob": ["1980-01-15", "1990-05-20", "1975-11-10", "1985-03-25", "1992-07-30"],
        "gender": ["M", "F", "M", "F", "M"],
        "admission_date": ["2023-01-10", "2023-01-15", "2023-01-20", "2023-01-25", "2023-02-01"],
        "discharge_date": ["2023-01-15", "2023-01-20", "2023-01-25", "2023-01-30", "2023-02-05"],
        "diagnosis_code": ["J45.909", "E11.9", "I10", "J44.1", "E10.9"],
        "diagnosis_description": ["Asthma", "Type 2 DM", "Hypertension", "COPD", "Type 1 DM"],
        "medication": ["Albuterol", "Metformin", "Lisinopril", "Tiotropium", "Insulin"],
        "lab_glucose": [180, 220, 160, 190, 250],
        "lab_hba1c": [7.2, 8.1, 6.8, 7.5, 9.0],
        "vital_bp_sys": [140, 135, 150, 145, 155],
        "vital_bp_dia": [90, 85, 95, 90, 100],
        "procedure_count": [2, 1, 3, 1, 2],
        "outcome_days": [5, 5, 5, 5, 4],
        "free_text_notes": ["Patient responded well", "Required follow-up", "Stable condition", "Improved significantly", "Monitoring needed"]
    }

    import tempfile

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as temp_csv:
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(temp_csv.name, index=False)
        temp_csv_path = temp_csv.name

    try:
        # Define the task
        task_text = "Identify patients with Type 2 diabetes and analyze glucose control factors"

        # Initialize selector with configuration
        config = {
            "privacy_mode": "strict",
            "keep_free_text": False,
            "max_rows_profile": 100,
            "max_final_columns": 50
        }

        selector = TaskDrivenColumnSelector(config=config)

        # Run the selection process
        result_paths = selector.run(
            input_csv_path=temp_csv_path,
            task_text=task_text
        )

        print("\nExample completed!")
        print(f"Filtered CSV: {result_paths['filtered_csv_path']}")
        print(f"Selection Report: {result_paths['selection_report_path']}")

        # Display the content of the selection report
        with open(result_paths['selection_report_path'], 'r') as f:
            report = json.load(f)
            print("\nSelection Report Summary:")
            print(f"- Final columns: {len(report['final_columns'])}")
            print(f"- Task type: {report['task_spec']['task_type']}")
            print(f"- Needed modalities: {report['task_spec']['need_modalities']}")
            if report['warnings']:
                print(f"- Warnings: {len(report['warnings'])}")

        return result_paths

    finally:
        # Clean up temporary file
        os.unlink(temp_csv_path)


def run_directory_example():
    """Example usage for processing CSV files from a directory"""
    import tempfile

    # Create sample CSV data
    sample_data = {
        "patient_id": ["P001", "P002", "P003"],
        "age": [45, 67, 32],
        "gender": ["M", "F", "F"],
        "diagnosis": ["Diabetes", "Hypertension", "Asthma"],
        "lab_value": [6.5, 140, 42],
        "medication": ["Metformin", "Lisinopril", "Albuterol"],
        "name": ["John Doe", "Jane Smith", "Alice Brown"]  # This will be flagged as PHI
    }

    # Create temporary directory and CSV file
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "input_cleaned_38.csv")
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(csv_path, index=False)

        print(f"Created sample CSV: {csv_path}")

        # Process the file from directory
        task_text = "Analyze patient data for diabetes-related factors"
        config = {"privacy_mode": "strict"}

        selector = TaskDrivenColumnSelector(config=config)

        result_paths = selector.run_on_directory_csv(
            directory_path=temp_dir,
            csv_pattern="input_cleaned_38.csv",
            task_text=task_text
        )

        print(f"\nDirectory example completed!")
        print(f"Filtered CSV: {result_paths['filtered_csv_path']}")
        print(f"Selection Report: {result_paths['selection_report_path']}")

        # Show results
        filtered_df = pd.read_csv(result_paths['filtered_csv_path'])
        print(f"Final columns: {list(filtered_df.columns)}")

        return result_paths


def run_current_liver_dataset():
    """Run the selector against the current liver dataset in the project root."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    input_csv_path = os.path.join(project_root, "liver_notes_extracted.csv")

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Current liver dataset not found: {input_csv_path}")

    task_text = (
        "Identify factors associated with mortality risk and length of stay "
        "among hospitalized patients with cirrhosis or hepatic failure."
    )
    config = {
        "privacy_mode": "strict",
        "keep_free_text": False,
        "max_rows_profile": 1000,
        "max_final_columns": 100,
    }

    print(f"Using input dataset: {input_csv_path}")
    print(f"Task: {task_text}")

    selector = TaskDrivenColumnSelector(config=config)
    result_paths = selector.run(
        input_csv_path=input_csv_path,
        task_text=task_text,
        output_dir=project_root,
    )

    print("\nCurrent liver dataset run completed!")
    print(f"Filtered CSV: {result_paths['filtered_csv_path']}")
    print(f"Selection Report: {result_paths['selection_report_path']}")
    return result_paths


if __name__ == "__main__":
    print("Running current liver dataset...")
    run_current_liver_dataset()
