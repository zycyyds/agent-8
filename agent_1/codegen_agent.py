import asyncio
import json
import os
import re
import sys
from pathlib import Path

import agentscope
from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import ToolResponse, execute_python_code, execute_shell_command, write_text_file

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_1.agentscope_tool_loader import load_toolkit_from_config
from configs.loader import get_agent_config

AGENT_CFG = get_agent_config("dataset_codegen_agent")
GENERATED_SCRIPT_PATH = Path("agent_1/generated_reorganizer.py")
OBSERVATION_RECORDS_PATH = Path("reorganized_output/_meta/records.json")
THINKING_BLOCK_PATTERN = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
PYTHON_CODE_BLOCK_PATTERN = re.compile(r"```python\s*([\s\S]*?)```", re.IGNORECASE)
FORBIDDEN_SCRIPT_CALL_PATTERNS = (
    "execute_python_code(",
    "execute_shell_command(",
    "write_text_file(",
    "persist_records(",
)
FORBIDDEN_LITERAL_ID_PATTERNS = (
    'os.path.join(output_root, "id"',
    "os.path.join(output_root, 'id'",
    'Path(output_root) / "id"',
    "Path(output_root) / 'id'",
)
BINARY_TARGET_SUFFIXES = (".jpg", ".jpeg", ".png", ".pdf", ".xlsx", ".xls", ".csv", ".tsv")


def remove_existing_generated_script(script_path: Path = GENERATED_SCRIPT_PATH) -> None:
    absolute_path = script_path if script_path.is_absolute() else PROJECT_ROOT / script_path
    if absolute_path.exists():
        absolute_path.unlink()


def initialize_records_file(output_root: Path) -> Path:
    records_path = output_root / "_meta" / "records.json"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = records_path.with_suffix(".json.tmp")
    tmp_path.write_text("[]\n", encoding="utf-8")
    tmp_path.replace(records_path)
    return records_path


def build_react_system_prompt() -> str:
    return """你是 Step1 异构数据集 ReAct 代码生成助手。
请使用中文交流。
你必须使用同一个 agent 分两阶段完成任务：先记录，再生成脚本。
第一阶段必须按顺序使用三个工具把结果写入 reorganized_output/_meta/records.json：
1. 先对每个文件调用 record_source_file(file_path)
2. 再调用 record_file_modality(file_path)
3. 如果文件是表格，再调用 record_table_split_strategy(file_path)
只有当所有文件都完成记录后，才能进入第二阶段。
第二阶段必须读取 reorganized_output/_meta/records.json，并根据其中记录生成执行脚本。
不要展示原始 thinking。
生成脚本内部禁止出现 execute_python_code、execute_shell_command、write_text_file、persist_records。
生成脚本不得忽略 records 中已有 observation，不得把所有图片硬编码为 figure。
最终输出契约必须满足 reorganized_output/<patient_id>/<modality>/<filename>。
最终回复必须包含：
1. 本轮说明
2. 工具调用摘要
3. 结果摘要
4. 一个 ```python fenced code block，内容是本轮完整可执行脚本
""".strip()


def build_react_round_prompt(input_path: str, attempt_index: int, previous_issues: list[str]) -> str:
    issue_lines = "无"
    if previous_issues:
        issue_lines = "\n".join(f"- {item}" for item in previous_issues)
    return f"""第 {attempt_index} 轮，请处理输入路径：{input_path}
本轮必须继续使用同一个 agent 分两阶段完成任务：先记录，再生成脚本。
请先按顺序调用 record_source_file、record_file_modality、record_table_split_strategy，把工具结果写入 records.json。
完成记录后，再读取 reorganized_output/_meta/records.json 生成脚本。
生成脚本时不得把所有图片硬编码为 figure。
若存在上轮未通过项，请优先修复：
{issue_lines}
""".strip()


def _extract_visible_text_from_blocks(content: object) -> str:
    if isinstance(content, list):
        visible_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                visible_parts.append(str(block.get("text", "")))
        return "\n".join(part for part in visible_parts if part).strip()
    return ""


def sanitize_agent_output(content: object) -> str:
    if isinstance(content, list):
        return _extract_visible_text_from_blocks(content)
    text = THINKING_BLOCK_PATTERN.sub("", str(content or ""))
    return text.strip()


class ThinkingSafeOpenAIChatFormatter(OpenAIChatFormatter):
    async def _format(self, msgs: list[Msg]) -> list[dict]:
        sanitized_msgs = [_strip_thinking_from_msg(msg) for msg in msgs]
        return await super()._format(sanitized_msgs)


def _strip_thinking_from_msg(msg: Msg) -> Msg:
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return msg
    content_blocks = [
        block
        for block in msg.get_content_blocks()
        if str(block.get("type") or "") != "thinking"
    ]
    sanitized_msg = Msg(
        name=msg.name,
        content=content_blocks,
        role=msg.role,
        metadata=msg.metadata,
        timestamp=msg.timestamp,
        invocation_id=msg.invocation_id,
    )
    sanitized_msg.id = msg.id
    return sanitized_msg


def serialize_runtime_result(value: object) -> object:
    if isinstance(value, ToolResponse):
        return {
            "id": value.id,
            "content": value.content,
            "metadata": value.metadata,
            "stream": value.stream,
            "is_last": value.is_last,
            "is_interrupted": value.is_interrupted,
        }
    if isinstance(value, dict):
        return {key: serialize_runtime_result(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_runtime_result(item) for item in value]
    if isinstance(value, tuple):
        return [serialize_runtime_result(item) for item in value]
    return value


def normalize_generated_script(script_content: str) -> str:
    normalized = (script_content or "").strip()
    if "\\n" in normalized and "\n" not in normalized:
        normalized = bytes(normalized, "utf-8").decode("unicode_escape")
    return normalized.strip()


def _looks_like_python_script(script_content: str) -> bool:
    script = (script_content or "").strip()
    if not script:
        return False
    python_markers = (
        "import ",
        "from ",
        "def ",
        "class ",
        "if __name__ ==",
        "for ",
        "while ",
        "with ",
        "try:",
        "Path(",
        "json.",
        "shutil.",
        "pd.",
        "print(",
    )
    return any(marker in script for marker in python_markers)


def extract_generated_script(response_text: str) -> str | None:
    match = PYTHON_CODE_BLOCK_PATTERN.search(response_text or "")
    if not match:
        return None
    script = normalize_generated_script(match.group(1))
    if not _looks_like_python_script(script):
        return None
    return script


def build_execute_script_command(script_path: Path = GENERATED_SCRIPT_PATH) -> str:
    absolute_script_path = PROJECT_ROOT / script_path
    return f'PYTHONPATH="{PROJECT_ROOT}" python "{absolute_script_path}"'


def _extract_tool_feedback(result: object) -> str:
    if isinstance(result, dict):
        content = result.get("content")
        if content is not None:
            return str(content)
        return str(result)
    return str(result)


def _extract_command_stderr(result: object) -> str | None:
    content = _extract_tool_feedback(result)
    stderr_match = re.search(r"<stderr>([\s\S]*?)</stderr>", content)
    if stderr_match:
        stderr = stderr_match.group(1).strip()
        return stderr or None
    return None


def _extract_command_returncode(result: object) -> int | None:
    content = _extract_tool_feedback(result)
    returncode_match = re.search(r"<returncode>(-?\d+)</returncode>", content)
    if returncode_match:
        return int(returncode_match.group(1))
    return None


def _build_execution_feedback_issues(execute_result: object) -> list[str]:
    issues: list[str] = []
    returncode = _extract_command_returncode(execute_result)
    stderr = _extract_command_stderr(execute_result)

    if returncode not in (None, 0):
        issues.append(f"执行脚本失败，returncode={returncode}")
    if stderr:
        issues.append(f"执行 stderr：{stderr[:500]}")
    return issues


def collect_visible_source_files(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    return [
        item
        for item in sorted(path.rglob("*"))
        if item.is_file() and not any(part.startswith(".") for part in item.parts) and "reorganized_output" not in item.parts
    ]


def _detect_json_dump_to_binary_target_issue(script_content: str) -> str | None:
    normalized = script_content or ""
    lines = normalized.splitlines()
    tracked_binary_vars: set[str] = set()

    for line in lines:
        for suffix in BINARY_TARGET_SUFFIXES:
            if suffix in line:
                assign_match = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=", line)
                if assign_match:
                    tracked_binary_vars.add(assign_match.group(1))
                if "open(" in line and ('"w"' in line or "'w'" in line):
                    return f"禁止把 record JSON 写入医学原始文件目标路径，例如 {suffix}"

    for index, line in enumerate(lines):
        if "json.dump(" not in line:
            continue
        window = "\n".join(lines[max(0, index - 3): index + 1])
        if any(f"open({var}" in window for var in tracked_binary_vars):
            return "禁止把 record JSON 写入医学原始文件目标路径，例如 .jpg"
        if "open(" in window and any(suffix in window for suffix in BINARY_TARGET_SUFFIXES):
            return "禁止把 record JSON 写入医学原始文件目标路径，例如 .jpg"
    return None


def validate_generated_script_contract(script_content: str) -> dict:
    issues: list[str] = []
    normalized = script_content or ""

    for forbidden_call in FORBIDDEN_SCRIPT_CALL_PATTERNS:
        if forbidden_call in normalized:
            issues.append(f"生成脚本禁止调用控制器工具：{forbidden_call[:-1]}")

    for forbidden_path in FORBIDDEN_LITERAL_ID_PATTERNS:
        if forbidden_path in normalized:
            issues.append("输出路径禁止创建字面目录 id，必须直接使用 patient_id 作为第一层目录")
            break

    json_dump_to_binary_issue = _detect_json_dump_to_binary_target_issue(normalized)
    if json_dump_to_binary_issue:
        issues.append(json_dump_to_binary_issue)

    return {"passed": len(issues) == 0, "issues": issues}


async def run_react_codegen_cycle(
    input_path: str,
    agent,
    emit=print,
    script_path: Path = GENERATED_SCRIPT_PATH,
    max_rounds: int = 1,
) -> dict:
    last_write_result = {"status": "SKIPPED", "content": "脚本尚未写入"}
    last_execute_result = {"status": "SKIPPED", "content": "脚本尚未执行"}

    output_root = PROJECT_ROOT / "reorganized_output"
    records_path = initialize_records_file(output_root)
    emit(f"records.json 已初始化：{records_path}")

    reply = await agent(
        Msg(
            name="user",
            role="user",
            content=build_react_round_prompt(input_path, 1, []),
        )
    )
    response_text = sanitize_agent_output(reply.content)
    summary_text = re.sub(PYTHON_CODE_BLOCK_PATTERN, "", response_text).strip()
    emit(f"第 1 轮\n{summary_text}")

    script_content = extract_generated_script(response_text)
    if not script_content:
        emit("脚本提取失败：缺少可执行脚本。")
        return {
            "status": "FAILED",
            "rounds_used": 1,
            "write_result": last_write_result,
            "execute_result": last_execute_result,
            "issues": ["模型回复里没有完整 python 代码块"],
        }

    static_contract_result = validate_generated_script_contract(script_content)
    if not static_contract_result.get("passed"):
        emit(f"脚本静态检查：{json.dumps(static_contract_result, ensure_ascii=False)}")
        return {
            "status": "FAILED",
            "rounds_used": 1,
            "write_result": last_write_result,
            "execute_result": last_execute_result,
            "issues": static_contract_result["issues"],
        }

    remove_existing_generated_script(script_path)
    last_write_result = await write_text_file(
        file_path=str(PROJECT_ROOT / script_path),
        content=script_content,
    )
    last_execute_result = await execute_shell_command(build_execute_script_command(script_path))
    execution_issues = _build_execution_feedback_issues(last_execute_result)

    if execution_issues:
        return {
            "status": "FAILED",
            "rounds_used": 1,
            "write_result": last_write_result,
            "execute_result": last_execute_result,
            "issues": execution_issues,
        }

    return {
        "status": "SUCCESS",
        "rounds_used": 1,
        "write_result": last_write_result,
        "execute_result": last_execute_result,
    }


async def create_codegen_agent(toolkit) -> ReActAgent:
    model = OpenAIChatModel(
        model_name=os.environ.get("MODEL_NAME", AGENT_CFG.get("model_name", "gpt-4.1-mini")),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        stream=True,
        client_kwargs={
            "base_url": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        },
        generate_kwargs={
            "temperature": AGENT_CFG.get("temperature", 0.0),
        },
    )
    agent = ReActAgent(
        name="Dataset Codegen Agent",
        sys_prompt=build_react_system_prompt(),
        model=model,
        formatter=ThinkingSafeOpenAIChatFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
        max_iters=20,
    )
    agent.set_console_output_enabled(False)
    return agent


async def main() -> None:
    agentscope.init(project="DatasetCodegenAgent", name="Dataset Codegen Agent")

    toolkit = load_toolkit_from_config(str(PROJECT_ROOT / "agent_1" / "agentscope_tools_dataset.json"))
    toolkit.register_tool_function(write_text_file)
    toolkit.register_tool_function(execute_python_code)
    toolkit.register_tool_function(execute_shell_command)

    agent = await create_codegen_agent(toolkit)

    user = UserAgent(name="User")
    msg = Msg(name="system", content="请输入待处理的数据目录或文件路径。", role="system")
    print(f"\n{agent.name}: {msg.content}")

    while True:
        try:
            msg = await user(msg)
            user_input = str(msg.content).strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            result = await run_react_codegen_cycle(
                input_path=user_input,
                agent=agent,
                emit=print,
            )
            print(json.dumps(serialize_runtime_result(result), ensure_ascii=False, indent=2))
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    asyncio.run(main())
