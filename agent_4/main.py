import asyncio
import ast
import importlib.util
import json
import os
import random
import re
import shutil
from contextlib import contextmanager

import pandas as pd
from dotenv import load_dotenv
from agentscope.message import Msg

try:
    from .agents.cleaner_designer_agent import create_cleaner_designer_agent
    from .prompts import cleaner_generation_prompt, medical_analysis_prompt
    from .validators.data_validator import DataValidator
except ImportError:
    from agents.cleaner_designer_agent import create_cleaner_designer_agent
    from prompts import cleaner_generation_prompt, medical_analysis_prompt
    from validators.data_validator import DataValidator


load_dotenv()

ALLOWED_IMPORTS = {"pandas", "os", "re", "agentscope.tool"}
SAMPLE_SIZE = 20
MAX_ANALYSIS_RETRIES = 3
MAX_GENERATION_ATTEMPTS = 4
MAX_RUNTIME_REPAIR_ATTEMPTS = 2


@contextmanager
def temporary_working_directory(path: str):
    previous_cwd = os.getcwd()
    os.makedirs(path, exist_ok=True)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous_cwd)


def get_module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_default_validation_config_path() -> str:
    return os.path.join(get_module_dir(), "validation_config.json")


def resolve_input_csv(data_path: str) -> str:
    data_path = os.path.abspath(data_path)
    if os.path.isfile(data_path):
        if not data_path.lower().endswith(".csv"):
            raise FileNotFoundError(f"输入文件不是 CSV: {data_path}")
        return data_path

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"输入路径不存在: {data_path}")

    candidate_paths: list[str] = []
    for root, _, files in os.walk(data_path):
        for name in files:
            if name.lower().endswith(".csv"):
                candidate_paths.append(os.path.join(root, name))

    if not candidate_paths:
        raise FileNotFoundError(f"在目录 {data_path} 中未找到 CSV 文件。")

    basename_to_paths: dict[str, list[str]] = {}
    for path in candidate_paths:
        basename_to_paths.setdefault(os.path.basename(path).lower(), []).append(path)

    for preferred_name in ("all_patients.csv", "input.csv"):
        if preferred_name in basename_to_paths:
            paths = basename_to_paths[preferred_name]
            paths.sort(key=os.path.getmtime, reverse=True)
            return paths[0]

    candidate_paths.sort(key=os.path.getmtime, reverse=True)
    return candidate_paths[0]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    return df


def safe_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name))
    return f"_{cleaned}" if not cleaned or cleaned.startswith(".") else cleaned


def save_json(path: str, data: dict) -> None:
    absolute_path = os.path.abspath(path)
    parent = os.path.dirname(absolute_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(absolute_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def initialize_knowledge_files(columns: list[str], human_path: str, derived_path: str) -> tuple[dict, dict]:
    human = {col: "" for col in columns}
    derived = {col: "" for col in columns}
    save_json(human_path, human)
    save_json(derived_path, derived)
    return human, derived


def sample_values(series: pd.Series) -> list[str]:
    values = [str(v).strip() for v in series.dropna() if str(v).strip()]
    unique = list(dict.fromkeys(values))
    if len(unique) <= SAMPLE_SIZE:
        return unique
    return random.sample(unique, SAMPLE_SIZE)


def infer_column_profile(column_name: str, values: list[str]) -> dict:
    lower = column_name.lower()

    def ratio(fn) -> float:
        return sum(1 for v in values if fn(v)) / len(values) if values else 0.0

    is_num = lambda v: bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", v))
    is_dt = lambda v: any(
        re.fullmatch(p, v)
        for p in [
            r"\d{4}-\d{2}-\d{2}",
            r"\d{4}/\d{2}/\d{2}",
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?",
            r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}(:\d{2})?",
        ]
    )

    if any(token in lower for token in ["id", "index", "编号", "序号"]):
        return {"column_type": "identifier", "reason": "列名像标识列"}
    if any(token in lower for token in ["time", "date", "日期", "时间"]) or ratio(is_dt) >= 0.6:
        return {"column_type": "datetime", "reason": "列名或样本像时间列"}
    if any(token in lower for token in ["code", "icd", "编码", "代码"]):
        return {"column_type": "code", "reason": "列名像编码列"}
    if ratio(is_num) >= 0.8:
        return {"column_type": "numeric", "reason": "大部分样本是数值"}
    if len(set(values[:20])) <= 10 and len(values) >= 3:
        return {"column_type": "categorical", "reason": "样本重复度高"}
    if ratio(lambda v: len(v) >= 40) >= 0.5:
        return {"column_type": "free_text", "reason": "样本较长"}
    return {"column_type": "text", "reason": "按普通文本列处理"}


def parse_cleaner_response(text: str) -> dict:
    md = re.search(r"===CLEANER_MD===\s*(.*?)\s*===END_CLEANER_MD===", text, re.DOTALL)
    py = re.search(r"===IMPLEMENTATION_PY===\s*(.*?)\s*===END_IMPLEMENTATION_PY===", text, re.DOTALL)
    if md and py:
        return {
            "cleaner_md": md.group(1).strip(),
            "implementation_py": strip_code_fence(py.group(1).strip()),
        }
    raise ValueError("未找到完整的 cleaner 分段输出。")


def strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def validate_cleaner_package(pkg: dict) -> list[str]:
    errors = []
    cleaner_md = pkg.get("cleaner_md", "")
    code = pkg.get("implementation_py", "")

    if not cleaner_md.strip():
        errors.append("缺少 cleaner_md。")
    elif len(cleaner_md) > 2500:
        errors.append("cleaner_md 过长。")

    if not code.strip():
        errors.append("缺少 implementation_py。")
        return errors

    try:
        tree = ast.parse(code)
        compile(code, "<cleaner>", "exec")
    except SyntaxError as e:
        errors.append(f"Python 语法错误: {e.msg} (line {e.lineno})")
        return errors

    has_entry = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    errors.append(f"导入了未允许模块: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in ALLOWED_IMPORTS:
                errors.append(f"from-import 了未允许模块: {node.module}")
        elif isinstance(node, ast.FunctionDef) and node.name == "data_cleaning":
            has_entry = True

    if not has_entry:
        errors.append("缺少 data_cleaning 函数。")
    if "ToolResponse" not in code:
        errors.append("未使用 ToolResponse。")
    return errors


def load_and_run_cleaner(impl_path: str, input_path: str, module_name: str) -> tuple[str | None, list[str]]:
    try:
        spec = importlib.util.spec_from_file_location(module_name, impl_path)
        if spec is None or spec.loader is None:
            return None, [f"无法加载 cleaner: {impl_path}"]
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "data_cleaning"):
            return None, ["缺少 data_cleaning 函数。"]
        response = module.data_cleaning(input_path)
        output_file = getattr(response, "metadata", {}).get("output_file")
        if not output_file:
            return None, ["未返回 metadata.output_file。"]
        if not os.path.exists(output_file):
            return None, [f"输出文件不存在: {output_file}"]
        pd.read_csv(output_file, dtype=str, keep_default_na=False, na_values=[""])
        return output_file, []
    except Exception as e:
        return None, [f"运行失败: {e}"]


def _extract_clean_and_raw_text(content) -> tuple[str, str]:
    if isinstance(content, list):
        clean_parts = []
        raw_parts = []
        for item in content:
            if not isinstance(item, dict):
                raw_parts.append(str(item))
                clean_parts.append(str(item))
                continue
            text = item.get("text") if isinstance(item.get("text"), str) else ""
            item_type = str(item.get("type") or "")
            if text:
                raw_parts.append(text)
            if item_type != "thinking" and text:
                clean_parts.append(text)
        return "".join(clean_parts).strip(), "".join(raw_parts).strip()

    text = str(content).strip()
    return text, text


async def call_agent(prompt: str) -> tuple[str, str]:
    agent = create_cleaner_designer_agent()
    resp = await agent(Msg("user", prompt, "user"))
    content = getattr(resp, "content", resp)
    clean_text, raw_text = _extract_clean_and_raw_text(content)
    return clean_text, raw_text


async def retry_analysis(prompt: str) -> str:
    last_error = None
    for _ in range(MAX_ANALYSIS_RETRIES):
        try:
            clean_text, _ = await call_agent(prompt)
            return clean_text
        except Exception as e:
            last_error = e
            await asyncio.sleep(2)
    raise last_error


def default_summary_for_profile(column: str, profile: dict, human: str) -> str:
    profile_type = profile.get("column_type")
    if human.strip():
        return human.strip()
    if profile_type in {"identifier", "code"}:
        return f"{column} 像标识/编码列，应保持原值语义，不删重、不补缺失、不重编号，只做轻量格式规范化。"
    if profile_type == "datetime":
        return f"{column} 像时间列，应统一时间格式，避免补造时间。"
    if profile_type == "numeric":
        return f"{column} 像数值列，应仅做安全格式标准化，不改变数值意义。"
    return ""


def build_retry_prompt(column: str, summary: str, cleaner_name: str, profile: dict, errors: list[str]) -> str:
    base = cleaner_generation_prompt(column, summary, cleaner_name, profile)
    error_text = "\n".join(f"- {error}" for error in errors[:6])
    return (
        f"{base}\n\n【修正要求】\n"
        "请根据以下错误重写 cleaner，仅输出规定分段格式。\n"
        f"{error_text}\n"
    )


async def build_cleaner(column: str, values: list[str], profile: dict, human: str, derived_path: str, cleaner_dir: str, csv_path: str) -> tuple[dict | None, list[str]]:
    summary = default_summary_for_profile(column, profile, human)
    if not summary:
        analysis_prompt = medical_analysis_prompt(column, values, profile, human_advice=human)
        summary = await retry_analysis(analysis_prompt)

    with open(derived_path, "r", encoding="utf-8") as f:
        derived = json.load(f)
    derived[column] = summary
    save_json(derived_path, derived)

    cleaner_name = safe_name(column)
    prompt = cleaner_generation_prompt(column, summary, cleaner_name, profile)
    last_errors = []

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        clean_text, raw_text = await call_agent(prompt)
        with open(os.path.join(cleaner_dir, f"RAW_RESPONSE_attempt_{attempt}.txt"), "w", encoding="utf-8") as f:
            f.write(raw_text)

        try:
            pkg = parse_cleaner_response(clean_text)
        except Exception as e:
            last_errors = [f"输出格式错误: {e}"]
            prompt = build_retry_prompt(column, summary, cleaner_name, profile, last_errors)
            continue

        last_errors = validate_cleaner_package(pkg)
        if last_errors:
            prompt = build_retry_prompt(column, summary, cleaner_name, profile, last_errors)
            continue

        impl_path = os.path.join(cleaner_dir, "implementation.py")
        md_path = os.path.join(cleaner_dir, "CLEANER.md")
        with open(impl_path, "w", encoding="utf-8") as f:
            f.write(pkg["implementation_py"])
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(pkg["cleaner_md"])

        probe_path = os.path.join(cleaner_dir, f"probe_{cleaner_name}.csv")
        shutil.copyfile(csv_path, probe_path)
        output_file, run_errors = load_and_run_cleaner(impl_path, probe_path, f"probe_{cleaner_name}_{attempt}")
        for path in [probe_path, output_file]:
            if path and os.path.exists(path):
                os.remove(path)
        if not run_errors:
            return {"cleaner_name": cleaner_name, "summary": summary, "profile": profile}, []

        last_errors = run_errors
        prompt = build_retry_prompt(column, summary, cleaner_name, profile, run_errors)

    return None, last_errors


async def execute_cleaners(csv_path: str, cleaners: dict, cleaner_root: str, output_dir: str) -> tuple[str, dict]:
    current = os.path.abspath(csv_path)
    failed = {}
    counter = 1
    source_stem = os.path.splitext(os.path.basename(csv_path))[0]

    for column, info in cleaners.items():
        cleaner_name = info["cleaner_name"]
        impl_path = os.path.join(cleaner_root, cleaner_name, "implementation.py")
        output_file = None
        errors = []

        for attempt in range(1, MAX_RUNTIME_REPAIR_ATTEMPTS + 2):
            output_file, errors = load_and_run_cleaner(impl_path, current, f"run_{cleaner_name}_{attempt}")
            if not errors:
                break
            prompt = build_retry_prompt(column, info["summary"], cleaner_name, info["profile"], errors)
            clean_text, raw_text = await call_agent(prompt)
            with open(os.path.join(cleaner_root, cleaner_name, f"RUNTIME_REPAIR_attempt_{attempt}.txt"), "w", encoding="utf-8") as f:
                f.write(raw_text)
            try:
                pkg = parse_cleaner_response(clean_text)
                pkg_errors = validate_cleaner_package(pkg)
                if pkg_errors:
                    errors = pkg_errors
                    continue
                with open(impl_path, "w", encoding="utf-8") as f:
                    f.write(pkg["implementation_py"])
                with open(os.path.join(cleaner_root, cleaner_name, "CLEANER.md"), "w", encoding="utf-8") as f:
                    f.write(pkg["cleaner_md"])
            except Exception as e:
                errors = [f"修复 cleaner 失败: {e}"]

        if errors:
            failed[column] = errors
            continue

        target = os.path.join(output_dir, f"{source_stem}_cleaned_{counter}.csv")
        if output_file != target:
            if os.path.exists(target):
                os.remove(target)
            os.rename(output_file, target)
        current = target
        counter += 1

    return current, failed


def format_run_summary(result: dict) -> str:
    if not result.get("success"):
        return (
            "Step4: 数据质量检测与自动修复失败。\n"
            f"输入目录: {result.get('input_dir', '')}\n"
            f"输入文件: {result.get('input_csv', '')}\n"
            f"错误: {result.get('error', 'unknown error')}"
        )

    return "\n".join(
        [
            "Step4: 数据质量检测与自动修复完成。",
            f"输入目录: {result.get('input_dir', '')}",
            f"输入文件: {result.get('input_csv', '')}",
            f"清洗结果: {result.get('final_output_csv', '')}",
            f"质量评分: {result.get('quality_score', 0.0):.2f}",
            f"验证报告: {result.get('validation_report_path', '')}",
            f"详细报告: {result.get('detailed_report_path', '')}",
            f"失败列记录: {result.get('failed_columns_path', '')}",
            f"成功生成 cleaner 数: {result.get('generated_cleaners', 0)}",
            f"生成失败列数: {len(result.get('generation_failures', {}))}",
            f"运行失败列数: {len(result.get('runtime_failures', {}))}",
        ]
    )


async def run_data_quality_repair(
    input_dir: str,
    workspace_dir: str,
    validation_config_path: str | None = None,
    verbose: bool = True,
) -> dict:
    input_dir = os.path.abspath(input_dir)
    workspace_dir = os.path.abspath(workspace_dir)
    validation_config_path = os.path.abspath(validation_config_path or get_default_validation_config_path())

    cleaner_root = os.path.join(workspace_dir, "cleaners")
    human_path = os.path.join(workspace_dir, "human_knowledge.json")
    derived_path = os.path.join(workspace_dir, "derived_knowledge.json")

    result = {
        "success": False,
        "input_dir": input_dir,
        "input_csv": "",
        "workspace_dir": workspace_dir,
        "final_output_csv": "",
        "validation_report_path": "",
        "detailed_report_path": "",
        "failed_columns_path": "",
        "quality_score": None,
        "generated_cleaners": 0,
        "generation_failures": {},
        "runtime_failures": {},
        "error": "",
    }

    try:
        os.makedirs(workspace_dir, exist_ok=True)
        csv_path = resolve_input_csv(input_dir)
        result["input_csv"] = csv_path

        if verbose:
            print(f"📄 输入文件: {csv_path}")

        with temporary_working_directory(workspace_dir):
            if os.path.exists(cleaner_root):
                shutil.rmtree(cleaner_root)
            os.makedirs(cleaner_root, exist_ok=True)

            df = normalize_columns(pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""]))
            human_knowledge, _ = initialize_knowledge_files(list(df.columns), human_path, derived_path)
            if verbose:
                print("🧠 已按当前数据集重建 human_knowledge.json")
                print("🧠 已初始化 derived_knowledge.json")

            cleaners = {}
            generation_failures = {}

            for column in df.columns:
                values = sample_values(df[column])
                if not values:
                    continue

                profile = infer_column_profile(column, values)
                cleaner_dir = os.path.join(cleaner_root, safe_name(column))
                os.makedirs(cleaner_dir, exist_ok=True)

                if verbose:
                    print(f"🔍 处理列: {column}")
                cleaner_info, errors = await build_cleaner(
                    column,
                    values,
                    profile,
                    human_knowledge.get(column, ""),
                    derived_path,
                    cleaner_dir,
                    csv_path,
                )
                if cleaner_info is None:
                    generation_failures[column] = errors
                    if verbose:
                        print(f"  ❌ cleaner 生成失败: {errors}")
                    continue
                cleaners[column] = cleaner_info
                if verbose:
                    print(f"  ✅ cleaner 已生成: {cleaner_info['cleaner_name']}")

            result["generated_cleaners"] = len(cleaners)
            result["generation_failures"] = generation_failures

            if not cleaners:
                result["error"] = "没有可执行 cleaner，程序退出。"
                return result

            final_output, runtime_failures = await execute_cleaners(
                csv_path=csv_path,
                cleaners=cleaners,
                cleaner_root=cleaner_root,
                output_dir=workspace_dir,
            )
            result["final_output_csv"] = final_output
            result["runtime_failures"] = runtime_failures
            if verbose:
                print(f"📄 最终输出文件: {final_output}")

            cleaned_df = pd.read_csv(final_output, dtype=str, keep_default_na=False, na_values=[""])
            validator = DataValidator(cleaned_df, config_path=validation_config_path)
            report = validator.run_comprehensive_validation()
            score = validator.generate_quality_score()
            detail = validator.generate_detailed_report()

            report_path = final_output.replace(".csv", "_validation_report.json")
            detail_path = final_output.replace(".csv", "_detailed_report.txt")
            failed_columns_path = final_output.replace(".csv", "_failed_columns.json")
            save_json(report_path, report)
            with open(detail_path, "w", encoding="utf-8") as f:
                f.write(detail)

            failed_columns = {
                "generation_failures": generation_failures,
                "runtime_failures": runtime_failures,
            }
            save_json(failed_columns_path, failed_columns)

            result["validation_report_path"] = report_path
            result["detailed_report_path"] = detail_path
            result["failed_columns_path"] = failed_columns_path
            result["quality_score"] = score
            result["success"] = True

            if verbose:
                print(f"📈 质量评分: {score:.2f}")
                print(f"📋 验证报告: {report_path}")
                print(f"📄 详细报告: {detail_path}")
                if generation_failures or runtime_failures:
                    print("⚠️ 存在失败列，已写入 failed_columns.json")

        return result
    except Exception as e:
        result["error"] = str(e)
        return result


async def main():
    module_dir = get_module_dir()
    result = await run_data_quality_repair(
        input_dir=os.path.join(module_dir, "data"),
        workspace_dir=module_dir,
        validation_config_path=os.path.join(module_dir, "validation_config.json"),
        verbose=True,
    )
    print(format_run_summary(result))


if __name__ == "__main__":
    asyncio.run(main())
