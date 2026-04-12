import asyncio
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from typing import Any

import pandas as pd

from agentscope.agent import ReActAgent
from agentscope.formatter import OllamaChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OllamaChatModel
from agentscope.tool import ToolResponse

from agent_1.agentscope_tool_loader import load_toolkit_from_config
from memory_agent.config import config
from memory_agent.memory_tool import get_playbook_context_data
from memory_supervisor import report_pipeline_result


_CONTEXT_PREVIEW_LIMIT = 300
_MESSAGE_TEXT_LIMIT = 1500
_TOOL_OUTPUT_LIMIT = 2000


class ThinkingSafeOllamaChatFormatter(OllamaChatFormatter):
    """在送入 Ollama formatter 前过滤 thinking block，避免告警日志。"""

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
    # Preserve the original message id so streaming console output can append
    # deltas instead of printing the full accumulated message repeatedly.
    sanitized_msg.id = msg.id
    return sanitized_msg


def register_no_thinking_print_hook(agent: ReActAgent) -> ReActAgent:
    def _hook(_agent: ReActAgent, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        msg = kwargs.get("msg")
        if isinstance(msg, Msg):
            kwargs["msg"] = _strip_thinking_from_msg(msg)
        return kwargs

    agent.register_instance_hook("pre_print", "strip_thinking_for_console", _hook)
    return agent


def _truncate_text(value: Any, limit: int) -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= limit:
        return text, False
    return text[:limit] + "...(truncated)", True


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    return str(value)


def _parse_json_text(value: Any) -> Any | None:
    try:
        return json.loads(str(value or ""))
    except Exception:
        return None


def _tool_payload_has_error(payload: Any) -> bool:
    if isinstance(payload, dict):
        if payload.get("error"):
            return True
        status = str(payload.get("status", "")).lower()
        if status in {"error", "failed"}:
            return True
    return False


def _tool_event_succeeded(event: dict[str, Any]) -> bool:
    if str(event.get("status") or "") != "succeeded":
        return False
    payload = _parse_json_text(event.get("tool_output", ""))
    return not _tool_payload_has_error(payload)


def _find_latest_tool_event(events: list[dict[str, Any]], tool_name: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if str(event.get("tool_name") or "") == tool_name:
            return event
    return None


def _load_tool_preset_kwargs(config_path: str, function_name: str) -> dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    for item in raw.get("tools", []):
        for fn in item.get("functions", []):
            if isinstance(fn, dict) and str(fn.get("name") or "") == function_name:
                return dict(fn.get("preset_kwargs") or {})
    return {}


def _classification_requires_validation(infer_payload: dict[str, Any], project_root: str) -> bool | None:
    classification = infer_payload.get("classification")
    if isinstance(classification, list):
        return any(
            isinstance(item, dict) and str(item.get("modality") or "") == "ocr+figure"
            for item in classification
        )

    classification_path = infer_payload.get("classification_results")
    if not classification_path:
        classification_path = os.path.join(project_root, "output", "classification_results.json")

    try:
        with open(classification_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, list):
        return None

    for record in data:
        for group in record.get("病历", []) if isinstance(record, dict) else []:
            if not isinstance(group, dict):
                continue
            for images in group.values():
                if not isinstance(images, list):
                    continue
                if any(isinstance(img, dict) and img.get("modality") == "ocr+figure" for img in images):
                    return True
    return False


def _invoke_organize_dataset_tool(toolkit_path: str, validated_json_path: str | None) -> tuple[dict[str, Any], str]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preset_kwargs = _load_tool_preset_kwargs(toolkit_path, "organize_dataset_by_modality")
    tool_input: dict[str, Any] = {
        "validated_json_path": validated_json_path,
        "output_root": os.path.join(project_root, "program", "output", "data"),
        "result_json_dir": os.path.join(project_root, "output", "result_json"),
        "rawdata_root": os.path.join(project_root, "rawdata"),
    }
    tool_input.update(preset_kwargs)
    tool_input = {key: value for key, value in tool_input.items() if value is not None}

    module = importlib.import_module("agent_1.layout_analysis_tool")
    tool_fn = getattr(module, "organize_dataset_by_modality")
    response = tool_fn(**tool_input)
    return tool_input, str(getattr(response, "content", response))


def _maybe_run_step1_organize_fallback(
    toolkit_path: str,
    full_tool_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    organize_event = _find_latest_tool_event(full_tool_events, "Organize Dataset By Modality")
    if organize_event and _tool_event_succeeded(organize_event):
        return None

    infer_event = _find_latest_tool_event(full_tool_events, "Infer And Save Layout")
    if not infer_event or not _tool_event_succeeded(infer_event):
        return None

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    infer_payload = _parse_json_text(infer_event.get("tool_output", ""))
    if not isinstance(infer_payload, dict):
        return None

    requires_validation = _classification_requires_validation(infer_payload, project_root)
    validated_json_path: str | None = None

    if requires_validation:
        validation_event = _find_latest_tool_event(full_tool_events, "prepare_and_launch_validation_gui")
        if not validation_event or not _tool_event_succeeded(validation_event):
            return None

        validation_payload = _parse_json_text(validation_event.get("tool_output", ""))
        if not isinstance(validation_payload, dict):
            return None

        validation_status = str(validation_payload.get("status") or "").lower()
        if validation_status not in {"success", "skipped"}:
            return None
        validated_json_path = str(validation_payload.get("output_file") or "").strip() or None
    elif requires_validation is None:
        return None

    tool_input, tool_output = _invoke_organize_dataset_tool(toolkit_path, validated_json_path)
    tool_payload = _parse_json_text(tool_output)
    failed = _tool_payload_has_error(tool_payload)
    return {
        "event_id": "fallback_organize_dataset_by_modality",
        "tool_name": "Organize Dataset By Modality",
        "tool_input": tool_input,
        "tool_output": tool_output,
        "status": "failed" if failed else "succeeded",
        "error": tool_output if failed else "",
    }


def _normalize_content_blocks(content: Any, truncate: bool = True) -> tuple[list[dict[str, Any]], list[str]]:
    truncated_flags: list[str] = []
    blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]
    normalized: list[dict[str, Any]] = []

    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            text = str(block or "")
            was_truncated = False
            if truncate:
                text, was_truncated = _truncate_text(block, _MESSAGE_TEXT_LIMIT)
            if was_truncated:
                truncated_flags.append(f"raw_block_{idx}")
            normalized.append({"type": "raw", "value": text})
            continue

        block_type = str(block.get("type") or "raw")
        item: dict[str, Any] = {"type": block_type}

        if block_type == "text":
            text = str(block.get("text", ""))
            was_truncated = False
            if truncate:
                text, was_truncated = _truncate_text(block.get("text", ""), _MESSAGE_TEXT_LIMIT)
            item["text"] = text
            if was_truncated:
                truncated_flags.append(f"text_{idx}")
        elif block_type == "thinking":
            continue
        elif block_type == "tool_use":
            item["id"] = str(block.get("id", ""))
            item["name"] = str(block.get("name", ""))
            item["input"] = _json_safe(block.get("input", {}))
        elif block_type == "tool_result":
            output = str(block.get("output", ""))
            was_truncated = False
            if truncate:
                output, was_truncated = _truncate_text(block.get("output", ""), _TOOL_OUTPUT_LIMIT)
            item["id"] = str(block.get("id", ""))
            item["name"] = str(block.get("name", ""))
            item["output"] = output
            if was_truncated:
                truncated_flags.append(f"tool_result_{idx}")
        else:
            item.update(_json_safe(block))

        normalized.append(item)

    return normalized, truncated_flags


def _extract_tool_events(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}

    def _is_error_output(text: str) -> bool:
        raw = str(text or "")
        return any(keyword in raw for keyword in ("Error:", "Traceback", "失败", "异常", "unexpected keyword"))

    for message in raw_messages:
        for block in message.get("content_blocks", []):
            block_type = block.get("type")
            if block_type == "tool_use":
                event = {
                    "tool_name": str(block.get("name", "")),
                    "tool_input": _json_safe(block.get("input", {})),
                    "tool_output": "",
                    "status": "requested",
                    "error": "",
                }
                event_id = str(block.get("id", ""))
                if event_id:
                    pending[event_id] = event
                events.append(event)
            elif block_type == "tool_result":
                output = str(block.get("output", ""))
                status = "failed" if _is_error_output(output) else "succeeded"
                error = output if status == "failed" else ""
                event_id = str(block.get("id", ""))
                if event_id and event_id in pending:
                    pending[event_id]["tool_output"] = output
                    pending[event_id]["status"] = status
                    pending[event_id]["error"] = error
                else:
                    events.append(
                        {
                            "tool_name": str(block.get("name", "")),
                            "tool_input": {},
                            "tool_output": output,
                            "status": status,
                            "error": error,
                        }
                    )

    return events


def _extract_assistant_outputs(raw_messages: list[dict[str, Any]]) -> list[str]:
    outputs: list[str] = []
    for message in raw_messages:
        if str(message.get("role") or "") != "assistant":
            continue
        for block in message.get("content_blocks", []):
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type == "text":
                text = str(block.get("text") or "").strip()
            elif block_type == "raw":
                text = str(block.get("value") or "").strip()
            else:
                text = ""
            if text:
                outputs.append(text)
    return outputs


def _format_message_blocks(content: Any) -> str:
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "thinking":
                    continue
                elif block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    text_parts.append(f"[Tool Use] {block.get('name')} with arguments: {block.get('input')}")
                elif block_type == "tool_result":
                    text_parts.append(f"[Tool Result] {block.get('name')}: {block.get('output')}")
                else:
                    text_parts.append(str(block))
            else:
                text_parts.append(str(block))
        return "\n".join(text_parts)
    return str(content)


class PipelineTraceCollector:
    """收集流水线每一步的完整执行痕迹，并导出结构化 payload。"""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def record_step(
        self,
        step_name: str,
        input_data: str,
        output_content: str,
        context: str = "",
        thinking_chain: list[dict] | None = None,
        raw_messages: list[dict] | None = None,
        full_raw_messages: list[dict] | None = None,
        tool_events: list[dict] | None = None,
        full_tool_events: list[dict] | None = None,
        truncated_flags: list[str] | None = None,
    ) -> None:
        self.steps.append({
            "step_name": step_name,
            "input_data": input_data,
            "context": context,
            "output_content": output_content,
            "thinking_chain": thinking_chain or [],
            "raw_messages": raw_messages or [],
            "full_raw_messages": full_raw_messages or [],
            "tool_events": tool_events or [],
            "full_tool_events": full_tool_events or [],
            "truncated_flags": truncated_flags or [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def build_trace_payload(
        self,
        input_text: str,
        duration_seconds: float,
        orchestrator_summary: str,
        retrieved_bullet_ids: list[str],
    ) -> dict[str, Any]:
        steps = []
        for step in self.steps:
            context_preview, context_truncated = _truncate_text(step.get("context", ""), _CONTEXT_PREVIEW_LIMIT)
            truncated_flags = list(step.get("truncated_flags", []))
            if context_truncated and "context_preview" not in truncated_flags:
                truncated_flags.append("context_preview")
            standard_tool_events = step.get("full_tool_events") or step.get("tool_events", [])
            preview_tool_events = step.get("tool_events", [])
            assistant_output_source = step.get("full_raw_messages") or step.get("raw_messages", [])
            steps.append(
                {
                    "step_name": step["step_name"],
                    "timestamp": step["timestamp"],
                    "input_data": step["input_data"],
                    "context_preview": context_preview,
                    "assistant_outputs": _json_safe(_extract_assistant_outputs(assistant_output_source)),
                    "tool_events": _json_safe(standard_tool_events),
                    "tool_events_preview": _json_safe(preview_tool_events),
                    "full_tool_events": _json_safe(step.get("full_tool_events", [])),
                    "final_output": step["output_content"],
                    "raw_messages": _json_safe(step.get("raw_messages", [])),
                    "full_raw_messages": _json_safe(step.get("full_raw_messages", [])),
                    "truncated_flags": _json_safe(truncated_flags),
                }
            )

        success = not any(
            str(event.get("status") or "") == "failed"
            for step in steps
            for event in step.get("tool_events", [])
            if isinstance(event, dict)
        )

        return {
            "input_text": str(input_text or ""),
            "duration_seconds": float(duration_seconds or 0.0),
            "orchestrator_summary": str(orchestrator_summary or ""),
            "retrieved_bullet_ids": [str(item) for item in (retrieved_bullet_ids or []) if str(item).strip()],
            "steps": steps,
            "raw_trace_text": self.build_full_trace(),
            "full_raw_trace_text": self.build_full_trace(full=True),
            "success": success,
        }

    def build_full_trace(self, full: bool = False) -> str:
        if not self.steps:
            return "(无步骤记录)"

        parts: list[str] = []
        for i, step in enumerate(self.steps, 1):
            parts.append(f"{'=' * 60}")
            parts.append(f"Step {i}: {step['step_name']}")
            parts.append(f"{'=' * 60}")
            parts.append(f"时间: {step['timestamp']}")
            parts.append(f"输入: {step['input_data']}")
            if step["context"]:
                ctx_preview = step["context"][:300]
                if len(step["context"]) > 300:
                    ctx_preview += "..."
                parts.append(f"上下文: {ctx_preview}")

            parts.append(f"最终输出:\n{step['output_content']}")
            parts.append("")
        return "\n".join(parts)

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)


_trace_collector = PipelineTraceCollector()


def get_trace_collector() -> PipelineTraceCollector:
    return _trace_collector


def build_single_step_trace_payload(
    step_index: int,
    input_text: str,
    duration_seconds: float,
    orchestrator_summary: str,
    retrieved_bullet_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    if not _trace_collector.steps or abs(step_index) > len(_trace_collector.steps):
        return None
    step = _trace_collector.steps[step_index]
    collector_cls = type(_trace_collector)
    temp_collector = collector_cls()
    temp_collector.steps = [step]
    return temp_collector.build_trace_payload(
        input_text=input_text,
        duration_seconds=duration_seconds,
        orchestrator_summary=orchestrator_summary,
        retrieved_bullet_ids=retrieved_bullet_ids or [],
    )


async def report_last_step_reflection(
    input_text: str,
    duration_seconds: float,
    orchestrator_summary: str,
    retrieved_bullet_ids: list[str] | None = None,
) -> str | None:
    payload = build_single_step_trace_payload(
        step_index=-1,
        input_text=input_text,
        duration_seconds=duration_seconds,
        orchestrator_summary=orchestrator_summary,
        retrieved_bullet_ids=retrieved_bullet_ids,
    )
    if not payload:
        return None
    return await report_pipeline_result(payload, used_bullet_ids=retrieved_bullet_ids)


def _get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_program_output_root() -> str:
    return os.path.join(_get_project_root(), "program", "output")


def _get_step5_results_dir() -> str:
    return os.path.join(_get_program_output_root(), "step5_results")


def _get_step67_results_dir() -> str:
    return os.path.join(_get_program_output_root(), "step6-7_results")


def _get_step67_step6_output_dir() -> str:
    return os.path.join(_get_step67_results_dir(), "step6")


def _get_step67_step7_output_dir() -> str:
    return os.path.join(_get_step67_results_dir(), "step7")


def _get_default_step2_3_input_path(input_data: str) -> str:
    if input_data and os.path.isdir(str(input_data)):
        return str(input_data)
    return os.path.join(_get_program_output_root(), "data")


def _get_default_step4_input_path(input_data: str) -> str:
    step2_3_root = os.path.join(_get_program_output_root(), "step2_3_results")
    if os.path.isdir(step2_3_root):
        return step2_3_root

    if input_data and os.path.isdir(str(input_data)):
        return str(input_data)

    return os.path.join(_get_program_output_root(), "data")


def _get_default_step5_input_csv() -> str:
    step4_results_dir = os.path.join(_get_program_output_root(), "step4_results")
    if not os.path.isdir(step4_results_dir):
        return os.path.join(step4_results_dir, "latest_cleaned_missing.csv")

    cleaned_candidates: list[str] = []
    for name in os.listdir(step4_results_dir):
        lower_name = name.lower()
        if not lower_name.endswith(".csv"):
            continue
        if "_cleaned_" not in lower_name:
            continue
        cleaned_candidates.append(os.path.join(step4_results_dir, name))

    if cleaned_candidates:
        cleaned_candidates.sort(key=os.path.getmtime, reverse=True)
        return cleaned_candidates[0]

    return os.path.join(step4_results_dir, "latest_cleaned_missing.csv")


def _get_default_step5_task_text() -> str:
    return (
        "Identify factors associated with mortality risk and length of stay "
        "among hospitalized patients with cirrhosis or hepatic failure."
    )


def _resolve_step2_3_cleaner_root() -> str:
    candidate_roots = [
        os.path.join(_get_project_root(), "agent_2-3"),
        os.path.join(
            _get_project_root(),
            "medical-data-pipeline-public",
            "step1_5_medical_data_cleaner",
        ),
    ]
    cleaner_root = next((path for path in candidate_roots if os.path.isdir(path)), "")
    if not cleaner_root:
        raise FileNotFoundError(
            "未找到 Step2_3 清洗目录，已尝试: "
            + " | ".join(candidate_roots)
        )
    return cleaner_root


def _load_step1_5_cleaner_components():
    cleaner_root = _resolve_step2_3_cleaner_root()

    if cleaner_root not in sys.path:
        sys.path.insert(0, cleaner_root)

    main_module = importlib.import_module("main")
    agent_module = importlib.import_module("agents.unified_processing_agent")
    return main_module.process_single_input, main_module.run_interactive_session, agent_module.UnifiedProcessingAgent


def _load_step2_3_liver_notes_components():
    cleaner_root = _resolve_step2_3_cleaner_root()

    module_path = os.path.join(cleaner_root, "process_liver_notes.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"未找到 liver notes 处理脚本: {module_path}")

    module_name = "_step2_3_process_liver_notes"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 liver notes 模块: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module.setup_liver_agents, module.batch_process_liver_notes


def _resolve_step2_3_liver_jsonl_path(data_dir: str) -> str:
    candidates = [
        os.path.join(data_dir, "table", "liver_patients_note", "liver_patients_note.jsonl"),
        os.path.join(data_dir, "liver_patients_note", "table", "liver_patients_note.jsonl"),
        os.path.join(data_dir, "table", "liver_patients_note.jsonl"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def _build_step2_3_liver_summary(data_dir: str, result_jsonl: str, result_csv: str, fields_json: str) -> str:
    lines = [
        "Step2_3: liver notes JSONL 专项抽取完成。",
        f"输入目录: {data_dir}",
        f"输出JSONL: {result_jsonl}",
    ]
    if result_csv:
        lines.append(f"输出CSV: {result_csv}")
    if fields_json:
        lines.append(f"字段分析: {fields_json}")
    return "\n".join(lines)


def _summarize_step2_3_event(event: dict[str, Any]) -> str:
    pieces = [f"event={event.get('event', 'unknown')}"]
    if event.get("mode"):
        pieces.append(f"mode={event['mode']}")
    if event.get("target"):
        pieces.append(f"target={event['target']}")
    elif event.get("input"):
        pieces.append(f"input={event['input']}")
    if event.get("data_type"):
        pieces.append(f"data_type={event['data_type']}")
    if event.get("error"):
        pieces.append(f"error={event['error']}")
    return " | ".join(str(item) for item in pieces if str(item).strip())


def _build_step2_3_trace_payload(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assistant_outputs: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    for index, event in enumerate(result.get("session_events", []), 1):
        if not isinstance(event, dict):
            continue
        assistant_outputs.append(
            {
                "role": "assistant",
                "name": f"step2_3_session_{index}",
                "content_blocks": [{"type": "text", "text": _summarize_step2_3_event(event)}],
            }
        )
        event_name = str(event.get("event") or "session_event")
        event_target = str(event.get("target") or event.get("input") or "")
        event_output_parts = []
        if event.get("data_type"):
            event_output_parts.append(f"data_type={event['data_type']}")
        if event.get("statistics"):
            event_output_parts.append(json.dumps(event["statistics"], ensure_ascii=False))
        if event.get("stats"):
            event_output_parts.append(json.dumps(event["stats"], ensure_ascii=False))
        if event.get("output_file"):
            event_output_parts.append(f"output_file={event['output_file']}")
        if event.get("error"):
            event_output_parts.append(f"error={event['error']}")
        tool_events.append(
            {
                "tool_name": f"step2_3_{event_name}",
                "tool_input": {"input": event_target, "mode": event.get("mode")},
                "tool_output": " | ".join(part for part in event_output_parts if part),
                "status": "succeeded" if event.get("success", False) else "failed",
                "error": str(event.get("error") or ""),
            }
        )
    return assistant_outputs, tool_events


def _build_step2_3_summary(data_dir: str, result: dict[str, Any]) -> str:
    if not result:
        return f"Step2_3: 未返回结果。输入目录: {data_dir}"

    if "quit_reason" in result:
        lines = [
            "Step2_3: 医学数据清洗与标准化交互会话结束。",
            f"输入目录: {data_dir}",
            f"退出方式: {result.get('quit_reason', 'unknown')}",
            f"处理轮次: {result.get('processed_inputs', 0)}",
            f"成功次数: {result.get('success_count', 0)}",
            f"失败次数: {result.get('failure_count', 0)}",
            f"切换分析模式次数: {result.get('analyze_requests', 0)}",
            f"查看统计次数: {result.get('stats_requests', 0)}",
        ]
        if result.get("last_input"):
            lines.append(f"最后一次输入: {result['last_input']}")
        if result.get("output_dir"):
            lines.append(f"输出目录: {result['output_dir']}")
        if result.get("last_error"):
            lines.append(f"最后错误: {result['last_error']}")
        return "\n".join(lines)

    if not result.get("success"):
        return (
            f"Step2_3: 医学数据清洗与标准化失败。\n"
            f"输入目录: {data_dir}\n"
            f"错误: {result.get('error', 'unknown error')}"
        )

    stats = result.get("statistics", {})
    lines = [
        "Step2_3: 医学数据清洗与标准化完成。",
        f"输入目录: {data_dir}",
        f"数据类型: {result.get('data_type', 'unknown')}",
    ]

    if result.get("data_type") == "directory":
        lines.extend(
            [
                f"总病人数: {stats.get('total_patients', 0)}",
                f"总文件数: {stats.get('total_files', 0)}",
                f"成功处理: {stats.get('processed_files', 0)}",
                f"处理失败: {stats.get('failed_files', 0)}",
            ]
        )
    elif stats:
        lines.extend(
            [
                f"总行数: {stats.get('total_rows', 'N/A')}",
                f"术语标准化成功数: {stats.get('terms_standardized', 0)}",
                f"量纲统一数: {stats.get('values_normalized', 0)}",
            ]
        )

    csv_results_folder = result.get("csv_results_folder")
    if csv_results_folder:
        lines.append(f"CSV结果目录: {csv_results_folder}")

    output_files = result.get("output_files")
    if isinstance(output_files, dict) and output_files:
        lines.append(f"输出文件数: {len(output_files)}")

    output_file = result.get("output_file")
    if output_file:
        lines.append(f"输出文件: {output_file}")

    return "\n".join(lines)


def _build_step4_summary(input_dir: str, result: dict[str, Any]) -> str:
    if not result:
        return f"Step4: 未返回结果。输入目录: {input_dir}"

    if not result.get("success"):
        return (
            f"Step4: 数据质量检测与自动修复失败。\n"
            f"输入目录: {input_dir}\n"
            f"预处理输入CSV: {result.get('prepared_input_csv', '')}\n"
            f"输入文件: {result.get('input_csv', '')}\n"
            f"错误: {result.get('error', 'unknown error')}"
        )

    lines = [
        "Step4: 数据质量检测与自动修复完成。",
        f"输入目录: {input_dir}",
        f"预处理输入CSV: {result.get('prepared_input_csv', '')}",
        f"输入文件: {result.get('input_csv', '')}",
        f"JSON 转换文件数: {int(result.get('prepared_json_count') or 0)}",
        f"清洗结果: {result.get('final_output_csv', '')}",
        f"质量评分: {float(result.get('quality_score') or 0.0):.2f}",
        f"验证报告: {result.get('validation_report_path', '')}",
        f"详细报告: {result.get('detailed_report_path', '')}",
        f"失败列记录: {result.get('failed_columns_path', '')}",
        f"成功生成 cleaner 数: {result.get('generated_cleaners', 0)}",
        f"生成失败列数: {len(result.get('generation_failures', {}) or {})}",
        f"运行失败列数: {len(result.get('runtime_failures', {}) or {})}",
    ]
    return "\n".join(lines)


def _build_step4_trace_payload(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_dir = str(result.get("input_dir") or "")
    input_csv = str(result.get("input_csv") or "")
    final_output_csv = str(result.get("final_output_csv") or "")
    quality_score = result.get("quality_score")
    generation_failures = result.get("generation_failures") or {}
    runtime_failures = result.get("runtime_failures") or {}
    prepared_csv = str(result.get("prepared_input_csv") or "")
    prepared_json_count = int(result.get("prepared_json_count") or 0)
    summary_text = (
        f"input_dir={input_dir} | input_csv={input_csv} | prepared_input_csv={prepared_csv} | final_output_csv={final_output_csv} | "
        f"quality_score={quality_score} | generated_cleaners={result.get('generated_cleaners', 0)} | "
        f"generation_failures={len(generation_failures)} | runtime_failures={len(runtime_failures)}"
    )
    assistant_outputs = [
        {
            "role": "assistant",
            "name": "step4_runner",
            "content_blocks": [{"type": "text", "text": summary_text}],
        }
    ]

    prepare_skipped_reuse_existing_csv = (not prepared_csv) and prepared_json_count == 0 and bool(input_csv)
    prepare_succeeded = bool(prepared_csv) or prepare_skipped_reuse_existing_csv
    prepare_output = f"prepared_input_csv={prepared_csv} | prepared_json_count={prepared_json_count}"
    if prepare_skipped_reuse_existing_csv:
        prepare_output = (
            f"prepared_input_csv=<skipped_reuse_existing_csv> | prepared_json_count=0 | input_csv={input_csv}"
        )

    tool_events = [
        {
            "tool_name": "step4_prepare_input",
            "tool_input": {"input_dir": input_dir},
            "tool_output": prepare_output,
            "status": "succeeded" if prepare_succeeded else "failed",
            "error": "" if prepare_succeeded else str(result.get('error') or "未生成可用的输入 CSV"),
        },
        {
            "tool_name": "step4_resolve_input_csv",
            "tool_input": {"input_dir": input_dir},
            "tool_output": f"input_csv={input_csv}",
            "status": "succeeded" if input_csv else "failed",
            "error": "" if input_csv else str(result.get('error') or "未找到输入 CSV"),
        },
        {
            "tool_name": "step4_generate_cleaners",
            "tool_input": {"input_csv": input_csv},
            "tool_output": (
                f"generated_cleaners={result.get('generated_cleaners', 0)} | "
                f"generation_failures={len(generation_failures)}"
            ),
            "status": "succeeded" if result.get("generated_cleaners", 0) > 0 else "failed",
            "error": "" if result.get("generated_cleaners", 0) > 0 else str(result.get('error') or "未生成可执行 cleaner"),
        },
        {
            "tool_name": "step4_execute_cleaners",
            "tool_input": {"input_csv": input_csv},
            "tool_output": f"final_output_csv={final_output_csv} | runtime_failures={len(runtime_failures)}",
            "status": "succeeded" if final_output_csv else "failed",
            "error": "" if final_output_csv else str(result.get('error') or "未生成清洗结果 CSV"),
        },
        {
            "tool_name": "step4_run_validation",
            "tool_input": {"final_output_csv": final_output_csv},
            "tool_output": (
                f"quality_score={quality_score} | validation_report={result.get('validation_report_path', '')} | "
                f"detailed_report={result.get('detailed_report_path', '')}"
            ),
            "status": "succeeded" if result.get("validation_report_path") else "failed",
            "error": "" if result.get("validation_report_path") else str(result.get('error') or "未生成验证报告"),
        },
        {
            "tool_name": "step4_write_reports",
            "tool_input": {"workspace_dir": result.get("workspace_dir", "")},
            "tool_output": f"failed_columns={result.get('failed_columns_path', '')}",
            "status": "succeeded" if result.get("failed_columns_path") else "failed",
            "error": "" if result.get("failed_columns_path") else str(result.get('error') or "未写入失败列记录"),
        },
    ]
    return assistant_outputs, tool_events


def _build_step5_summary(input_csv: str, task_text: str, result: dict[str, Any]) -> str:
    if not result:
        return f"Step5: 未返回结果。输入文件: {input_csv}"

    if not result.get("success"):
        return (
            f"Step5: 任务导向裁剪失败。\n"
            f"输入文件: {input_csv}\n"
            f"任务描述: {task_text}\n"
            f"错误: {result.get('error', 'unknown error')}"
        )

    return "\n".join(
        [
            "Step5: 任务导向裁剪完成。",
            f"输入文件: {input_csv}",
            f"任务描述: {task_text}",
            f"筛选结果: {result.get('filtered_csv_path', '')}",
            f"选择报告: {result.get('selection_report_path', '')}",
        ]
    )


_step67_runtime_cache: dict[str, Any] = {}


def _load_step67_pipeline_module():
    module_name = "agent_6_7_runtime_pipeline"
    module_path = os.path.join(_get_project_root(), "agent_6-7", "run_step6_7_ml_pipeline.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"未找到 Step6_7 管道文件: {module_path}")
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 Step6_7 管道模块: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def _resolve_latest_step5_artifacts() -> tuple[str, str]:
    step5_dir = _get_step5_results_dir()
    if not os.path.isdir(step5_dir):
        return "", ""

    filtered_candidates = [
        os.path.join(step5_dir, name)
        for name in os.listdir(step5_dir)
        if name.lower().endswith(".csv") and "_filtered_" in name.lower()
    ]
    if not filtered_candidates:
        return "", ""

    filtered_candidates.sort(key=os.path.getmtime, reverse=True)
    filtered_csv = filtered_candidates[0]

    base_prefix = os.path.basename(filtered_csv)
    marker = "_filtered_"
    idx = base_prefix.lower().find(marker)
    report_prefix = base_prefix[:idx] if idx >= 0 else ""

    report_candidates = [
        os.path.join(step5_dir, name)
        for name in os.listdir(step5_dir)
        if name.lower().endswith(".json")
        and "_selection_report_" in name.lower()
        and (not report_prefix or name.startswith(report_prefix + "_selection_report_"))
    ]
    report_candidates.sort(key=os.path.getmtime, reverse=True)
    selection_report = report_candidates[0] if report_candidates else ""
    return filtered_csv, selection_report


def _build_step6_summary(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return (
            "Step6: 一致性验证失败。\n"
            f"输入CSV: {result.get('filtered_csv', '')}\n"
            f"错误: {result.get('error', 'unknown error')}"
        )

    return "\n".join(
        [
            "Step6: 一致性验证完成。",
            f"输入CSV: {result.get('filtered_csv', '')}",
            f"通过患者ID数: {len(result.get('passed_patient_ids') or [])}",
            f"Step6文本报告: {result.get('step6_report_txt', '')}",
            f"Step6结构化报告: {result.get('step6_report_json', '')}",
        ]
    )


def _build_step6_trace_payload(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    success = bool(result.get("success"))
    summary_text = (
        f"filtered_csv={result.get('filtered_csv', '')} | selection_report={result.get('selection_report', '')} | "
        f"passed_patients={len(result.get('passed_patient_ids') or [])} | success={success}"
    )
    assistant_outputs = [
        {
            "role": "assistant",
            "name": "step6_runner",
            "content_blocks": [{"type": "text", "text": summary_text}],
        }
    ]
    tool_events = [
        {
            "tool_name": "step6_resolve_step5_inputs",
            "tool_input": {"step5_results_dir": _get_step5_results_dir()},
            "tool_output": f"filtered_csv={result.get('filtered_csv', '')} | selection_report={result.get('selection_report', '')}",
            "status": "succeeded" if result.get("filtered_csv") else "failed",
            "error": "" if result.get("filtered_csv") else str(result.get("error") or "未找到 Step5 输出 CSV"),
        },
        {
            "tool_name": "step6_run_pipeline",
            "tool_input": {"output_step6": result.get("output_step6", "")},
            "tool_output": f"step6_report_txt={result.get('step6_report_txt', '')} | step6_report_json={result.get('step6_report_json', '')}",
            "status": "succeeded" if success else "failed",
            "error": "" if success else str(result.get("error") or "Step6 执行失败"),
        },
    ]
    return assistant_outputs, tool_events


def _build_step7_summary(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return (
            "Step7: 表型/知识确认失败。\n"
            f"输入CSV: {result.get('filtered_csv', '')}\n"
            f"错误: {result.get('error', 'unknown error')}"
        )

    return "\n".join(
        [
            "Step7: 表型/知识确认完成。",
            f"输入CSV: {result.get('filtered_csv', '')}",
            f"数据集CSV: {result.get('step7_dataset_csv', '')}",
            f"数据集JSONL: {result.get('step7_dataset_jsonl', '')}",
            f"数据集JSON: {result.get('step7_dataset_json', '')}",
        ]
    )


def _build_step7_trace_payload(result: dict[str, Any], step6_rerun: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    success = bool(result.get("success"))
    summary_text = (
        f"filtered_csv={result.get('filtered_csv', '')} | step6_rerun={step6_rerun} | "
        f"dataset_csv={result.get('step7_dataset_csv', '')} | success={success}"
    )
    assistant_outputs = [
        {
            "role": "assistant",
            "name": "step7_runner",
            "content_blocks": [{"type": "text", "text": summary_text}],
        }
    ]
    tool_events = [
        {
            "tool_name": "step7_load_step6_context",
            "tool_input": {},
            "tool_output": f"passed_patient_ids={result.get('passed_patient_ids', [])} | step6_rerun={step6_rerun}",
            "status": "succeeded" if result.get("passed_patient_ids") is not None else "failed",
            "error": "",
        },
        {
            "tool_name": "step7_run_pipeline",
            "tool_input": {"output_step7": result.get("output_step7", "")},
            "tool_output": (
                f"dataset_csv={result.get('step7_dataset_csv', '')} | "
                f"dataset_jsonl={result.get('step7_dataset_jsonl', '')} | "
                f"dataset_json={result.get('step7_dataset_json', '')}"
            ),
            "status": "succeeded" if success else "failed",
            "error": "" if success else str(result.get("error") or "Step7 执行失败"),
        },
    ]
    return assistant_outputs, tool_events


def _build_step5_trace_payload(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_csv = str(result.get("input_csv") or "")
    task_text = str(result.get("task_text") or "")
    filtered_csv_path = str(result.get("filtered_csv_path") or "")
    selection_report_path = str(result.get("selection_report_path") or "")
    success = bool(result.get("success"))

    summary_text = (
        f"input_csv={input_csv} | task_text={task_text} | filtered_csv_path={filtered_csv_path} | "
        f"selection_report_path={selection_report_path} | success={success}"
    )
    assistant_outputs = [
        {
            "role": "assistant",
            "name": "step5_runner",
            "content_blocks": [{"type": "text", "text": summary_text}],
        }
    ]

    run_error = "" if success else str(result.get("error") or "Step5 执行失败")
    resolve_error = "" if input_csv else str(result.get("error") or "未找到 Step5 输入 CSV")
    output_error = "" if (filtered_csv_path and selection_report_path) else str(result.get("error") or "未生成 Step5 输出文件")

    tool_events = [
        {
            "tool_name": "step5_resolve_input_csv",
            "tool_input": {},
            "tool_output": f"input_csv={input_csv}",
            "status": "succeeded" if input_csv else "failed",
            "error": resolve_error,
        },
        {
            "tool_name": "step5_run_task_driven_selector",
            "tool_input": {"input_csv": input_csv, "task_text": task_text},
            "tool_output": f"success={success}",
            "status": "succeeded" if success else "failed",
            "error": run_error,
        },
        {
            "tool_name": "step5_write_outputs",
            "tool_input": {"input_csv": input_csv},
            "tool_output": (
                f"filtered_csv_path={filtered_csv_path} | selection_report_path={selection_report_path}"
            ),
            "status": "succeeded" if (filtered_csv_path and selection_report_path) else "failed",
            "error": output_error,
        },
    ]
    return assistant_outputs, tool_events


def _extract_step4_row_from_json_payload(json_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    standardized = payload.get("standardized_result") if isinstance(payload.get("standardized_result"), dict) else {}
    extraction = payload.get("extraction_result") if isinstance(payload.get("extraction_result"), dict) else {}
    entities = standardized.get("entities") or extraction.get("entities") or []
    temporal_info = standardized.get("temporal_info") or extraction.get("temporal_info") or []
    quantity_info = standardized.get("quantity_info") or extraction.get("quantity_info") or []
    relations = standardized.get("relations") or extraction.get("relations") or []

    row: dict[str, Any] = {
        "source_json": json_path,
        "success": payload.get("success"),
        "data_type": payload.get("data_type"),
        "processing_stages": json.dumps(payload.get("processing_stages") or [], ensure_ascii=False),
        "preprocessed_text": payload.get("preprocessed_text"),
        "entity_count": len(entities) if isinstance(entities, list) else 0,
        "temporal_count": len(temporal_info) if isinstance(temporal_info, list) else 0,
        "quantity_count": len(quantity_info) if isinstance(quantity_info, list) else 0,
        "relation_count": len(relations) if isinstance(relations, list) else 0,
        "entities_summary": json.dumps(entities, ensure_ascii=False),
        "temporal_info": json.dumps(temporal_info, ensure_ascii=False),
        "quantity_info": json.dumps(quantity_info, ensure_ascii=False),
        "relations": json.dumps(relations, ensure_ascii=False),
        "impression": standardized.get("impression") or extraction.get("impression"),
        "indication": standardized.get("indication") or extraction.get("indication"),
        "standardization_stats": json.dumps(standardized.get("_standardization_stats") or {}, ensure_ascii=False),
    }

    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            category = str(entity.get("category") or "Other")
            name = str(entity.get("name") or "")
            if not name:
                continue
            safe_key = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_]+", "_", name).strip("_") or "unknown"
            prefix = f"{category}_{safe_key}"
            value = entity.get("normalized_value")
            if value is None:
                value = entity.get("value")
            row[f"{prefix}_value"] = value
            row[f"{prefix}_unit"] = entity.get("normalized_unit") or entity.get("unit")
            row[f"{prefix}_original_text"] = entity.get("original_text")

    return row


def _prepare_step4_input_dir(input_dir: str, workspace_dir: str) -> tuple[str, int]:
    candidate_csv: list[str] = []
    candidate_json: list[str] = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            lower_name = name.lower()
            absolute_path = os.path.join(root, name)
            if lower_name.endswith(".csv"):
                candidate_csv.append(absolute_path)
            elif lower_name.endswith(".json"):
                candidate_json.append(absolute_path)

    if candidate_csv:
        return input_dir, 0

    if not candidate_json:
        raise FileNotFoundError(f"在目录 {input_dir} 中未找到可用于 Step4 的 CSV 或 JSON 文件。")

    rows = []
    for json_path in sorted(candidate_json):
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            continue
        rows.append(_extract_step4_row_from_json_payload(json_path, payload))

    if not rows:
        raise ValueError(f"目录 {input_dir} 中的 JSON 无法转换为 Step4 输入表格。")

    os.makedirs(workspace_dir, exist_ok=True)
    prepared_csv = os.path.join(workspace_dir, "prepared_step4_input.csv")
    pd.DataFrame(rows).to_csv(prepared_csv, index=False)
    return prepared_csv, len(candidate_json)


async def run_step1_modal_recognition(input_data: str, context: str = "") -> ToolResponse:
    toolkit_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "agent_1",
        "agentscope_tools.json",
    )
    toolkit = load_toolkit_from_config(toolkit_path)

    base_sys_prompt = (
        "你是一个布局分析助手。你只能使用工具完成图片布局分析、人工验证和数据集整理。并且保持和用户用中文交流。\n"
        "【执行步骤】请严格按照以下步骤并必须只使用标准 JSON 格式调用所有工具，绝对禁止使用 `<tool_call>` 等 HTML/XML 标签。\n"
        "1）优先调用 `Collect Image Files` 工具获取待处理的图片列表及 input_path；\n"
        "2）调用 `Infer And Save Layout` 工具进行版面分析（执行后会自动更新分类结果）；\n"
        "3）检查第 2 步返回的结果，如有 `ocr+figure` 模态，调用 `prepare_and_launch_validation_gui` 验证；\n"
        "4）调用 `Organize Dataset By Modality` 整理数据集（必须传入 with_segmentation=True 参数）；\n"
        "5）最后输出最终报告，且所有面向用户的总结、追问、报错都必须使用中文。"
    )

    if not context:
        context, _ = get_playbook_context_data(query_text=input_data)

    full_sys_prompt = f"{base_sys_prompt}\n\n{context}" if context else base_sys_prompt

    model = OllamaChatModel(
        model_name=config.LLM_MODEL,
        options={
            "temperature": config.LLM_TEMPERATURE,
            "seed": config.LLM_SEED,
        },
    )
    agent = ReActAgent(
        name="Agent-1-Modal-Identification",
        sys_prompt=full_sys_prompt,
        model=model,
        formatter=ThinkingSafeOllamaChatFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
        max_iters=20,
    )

    msg = Msg(name="orchestrator", content=input_data, role="user")
    result_msg = await agent(msg)

    output_str = _format_message_blocks(result_msg.content)

    thinking_chain = []
    raw_messages = []
    full_raw_messages = []
    truncated_flags: list[str] = []
    try:
        for mem_msg in await agent.memory.get_memory():
            normalized_blocks, block_flags = _normalize_content_blocks(getattr(mem_msg, "content", ""))
            full_blocks, _ = _normalize_content_blocks(getattr(mem_msg, "content", ""), truncate=False)
            raw_messages.append({
                "role": getattr(mem_msg, "role", "unknown"),
                "name": getattr(mem_msg, "name", ""),
                "content_blocks": normalized_blocks,
            })
            full_raw_messages.append({
                "role": getattr(mem_msg, "role", "unknown"),
                "name": getattr(mem_msg, "name", ""),
                "content_blocks": full_blocks,
            })
            thinking_chain.append({
                "role": getattr(mem_msg, "role", "unknown"),
                "name": getattr(mem_msg, "name", ""),
                "content": _format_message_blocks(getattr(mem_msg, "content", "")),
            })
            truncated_flags.extend(block_flags)
    except Exception as e:
        thinking_chain.append({"role": "error", "name": "extract_failed", "content": str(e)})
        raw_messages.append({
            "role": "error",
            "name": "extract_failed",
            "content_blocks": [{"type": "text", "text": str(e)}],
        })
        full_raw_messages.append({
            "role": "error",
            "name": "extract_failed",
            "content_blocks": [{"type": "text", "text": str(e)}],
        })
        truncated_flags.append("memory_extract_failed")

    tool_events = _extract_tool_events(raw_messages)
    full_tool_events = _extract_tool_events(full_raw_messages)

    fallback_event = _maybe_run_step1_organize_fallback(toolkit_path, full_tool_events)
    if fallback_event:
        fallback_note = "检测到模型未真正执行 `Organize Dataset By Modality`，包装层已按固定流程补跑数据整理。"
        assistant_blocks = [
            {"type": "text", "text": fallback_note},
            {
                "type": "tool_use",
                "id": fallback_event["event_id"],
                "name": fallback_event["tool_name"],
                "input": fallback_event["tool_input"],
            },
        ]
        system_blocks = [
            {
                "type": "tool_result",
                "id": fallback_event["event_id"],
                "name": fallback_event["tool_name"],
                "output": fallback_event["tool_output"],
            }
        ]

        normalized_assistant, assistant_flags = _normalize_content_blocks(assistant_blocks)
        normalized_system, system_flags = _normalize_content_blocks(system_blocks)
        full_assistant, _ = _normalize_content_blocks(assistant_blocks, truncate=False)
        full_system, _ = _normalize_content_blocks(system_blocks, truncate=False)

        raw_messages.extend(
            [
                {"role": "assistant", "name": "step1_wrapper", "content_blocks": normalized_assistant},
                {"role": "system", "name": "step1_wrapper", "content_blocks": normalized_system},
            ]
        )
        full_raw_messages.extend(
            [
                {"role": "assistant", "name": "step1_wrapper", "content_blocks": full_assistant},
                {"role": "system", "name": "step1_wrapper", "content_blocks": full_system},
            ]
        )
        thinking_chain.extend(
            [
                {
                    "role": "assistant",
                    "name": "step1_wrapper",
                    "content": _format_message_blocks(assistant_blocks),
                },
                {
                    "role": "system",
                    "name": "step1_wrapper",
                    "content": _format_message_blocks(system_blocks),
                },
            ]
        )
        truncated_flags.extend(assistant_flags)
        truncated_flags.extend(system_flags)

        tool_events = _extract_tool_events(raw_messages)
        full_tool_events = _extract_tool_events(full_raw_messages)
        output_str = f"{output_str}\n\n[系统兜底]\n{fallback_note}\n{fallback_event['tool_output']}"

    _trace_collector.record_step(
        step_name="数据感知与模态识别",
        input_data=input_data,
        output_content=output_str,
        context=context,
        thinking_chain=thinking_chain,
        raw_messages=raw_messages,
        full_raw_messages=full_raw_messages,
        tool_events=tool_events,
        full_tool_events=full_tool_events,
        truncated_flags=truncated_flags,
    )
    memory_feedback = await report_last_step_reflection(
        input_text=input_data,
        duration_seconds=0.0,
        orchestrator_summary=output_str,
        retrieved_bullet_ids=[],
    )
    if memory_feedback:
        print(f"\n[Memory Supervisor Step1 反思结果]\n{memory_feedback}\n")

    return ToolResponse(content=output_str)


async def run_step2_parse_extract(input_data: str, context: str = "") -> ToolResponse:
    output = f"Step 2 (Mock): 收到数据 [{input_data[:50]}...] 并完成解析提取。"
    _trace_collector.record_step(
        step_name="解析与结构化抽取",
        input_data=input_data,
        output_content=output,
        context=context,
    )
    return ToolResponse(content=output)


async def run_step2_3_medical_data_cleaner(input_data: str, context: str = "") -> ToolResponse:
    data_dir = _get_default_step2_3_input_path(input_data)
    print(f"[Step2_3] 使用输入目录: {data_dir}")
    project_root = _get_project_root()
    output_dir = os.path.join(project_root, "program", "output", "step2_3_results")

    if not os.path.isdir(data_dir):
        output = f"Step2_3: 输入目录不存在，无法执行医学数据清洗与标准化。输入目录: {data_dir}"
        _trace_collector.record_step(
            step_name="Step2_3 医学数据清洗与标准化",
            input_data=data_dir,
            output_content=output,
            context=context,
        )
        memory_feedback = await report_last_step_reflection(
            input_text=data_dir,
            duration_seconds=0.0,
            orchestrator_summary=output,
            retrieved_bullet_ids=[],
        )
        if memory_feedback:
            print(f"\n[Memory Supervisor Step2_3 反思结果]\n{memory_feedback}\n")
        return ToolResponse(content=output)

    started_at = time.time()
    session_messages: list[dict[str, Any]] = []
    session_tool_events: list[dict[str, Any]] = []
    try:
        liver_jsonl_path = _resolve_step2_3_liver_jsonl_path(data_dir)
        if liver_jsonl_path:
            print(f"[Step2_3] 检测到 liver jsonl，启用专项处理: {liver_jsonl_path}")
            setup_liver_agents, batch_process_liver_notes = _load_step2_3_liver_notes_components()
            med_agent, std_agent = setup_liver_agents()
            if med_agent is None:
                raise RuntimeError("未能初始化 liver notes 抽取 Agent（请检查 OPENAI_API_KEY / MODEL_NAME / OPENAI_API_BASE 配置）")

            liver_output_dir = os.path.join(output_dir, "liver_notes")
            liver_out_jsonl = await batch_process_liver_notes(
                jsonl_path=liver_jsonl_path,
                med_agent=med_agent,
                std_agent=std_agent,
                output_dir=liver_output_dir,
                max_records=None,
                verbose=True,
            )
            if not liver_out_jsonl or not os.path.isfile(liver_out_jsonl):
                raise FileNotFoundError(f"liver notes 处理未生成 JSONL 输出: {liver_out_jsonl}")

            liver_out_csv = liver_out_jsonl.replace(".jsonl", ".csv")
            liver_fields_json = ""
            liver_basename = os.path.basename(liver_out_jsonl)
            ts_match = re.search(r"liver_notes_extracted_(\d{8}_\d{6})\.jsonl$", liver_basename)
            if ts_match:
                candidate_fields = os.path.join(liver_output_dir, f"extraction_fields_{ts_match.group(1)}.json")
                if os.path.isfile(candidate_fields):
                    liver_fields_json = candidate_fields

            output = _build_step2_3_liver_summary(
                data_dir=data_dir,
                result_jsonl=liver_out_jsonl,
                result_csv=liver_out_csv if os.path.isfile(liver_out_csv) else "",
                fields_json=liver_fields_json,
            )
            session_messages = [
                {
                    "role": "assistant",
                    "name": "step2_3_liver_notes",
                    "content_blocks": [
                        {
                            "type": "text",
                            "text": (
                                f"mode=liver_notes | input={liver_jsonl_path} | "
                                f"output_jsonl={liver_out_jsonl}"
                            ),
                        }
                    ],
                }
            ]
            session_tool_events = [
                {
                    "tool_name": "step2_3_liver_notes_batch_process",
                    "tool_input": {"jsonl_path": liver_jsonl_path, "output_dir": liver_output_dir},
                    "tool_output": output,
                    "status": "succeeded",
                    "error": "",
                }
            ]
        else:
            if not sys.stdin or not sys.stdin.isatty():
                output = (
                    "Step2_3: 当前环境不支持交互式医学数据清洗。\n"
                    f"输入目录: {data_dir}\n"
                    "原因: 标准输入不是交互终端（non-TTY）。"
                )
            else:
                _, run_interactive_session, UnifiedProcessingAgent = _load_step1_5_cleaner_components()
                agent = UnifiedProcessingAgent(
                    name="Step2_3MedicalCleaner",
                    use_llm=True,
                    use_umls=True,
                    verbose=False,
                )
                print("\n[Step2_3] 已进入交互模式。输入 'quit' 结束当前 Step2_3 并返回主编排器。")
                result = await run_interactive_session(
                    agent=agent,
                    output_dir=output_dir,
                    show_banner=True,
                )
                output = _build_step2_3_summary(data_dir, result)
                session_messages, session_tool_events = _build_step2_3_trace_payload(result)
    except Exception as e:
        output = (
            f"Step2_3: 医学数据清洗与标准化执行失败。\n"
            f"输入目录: {data_dir}\n错误: {e}"
        )

    _trace_collector.record_step(
        step_name="Step2_3 医学数据清洗与标准化",
        input_data=data_dir,
        output_content=output,
        context=context,
        raw_messages=session_messages,
        full_raw_messages=session_messages,
        tool_events=session_tool_events,
        full_tool_events=session_tool_events,
    )
    duration = time.time() - started_at
    memory_feedback = await report_last_step_reflection(
        input_text=data_dir,
        duration_seconds=duration,
        orchestrator_summary=output,
        retrieved_bullet_ids=[],
    )
    if memory_feedback:
        print(f"\n[Memory Supervisor Step2_3 反思结果]\n{memory_feedback}\n")
    return ToolResponse(content=output)


async def run_step3_semantic_standardization(input_data: str, context: str = "") -> ToolResponse:
    output = "Step 3 (Mock): 完成标准化与映射。"
    _trace_collector.record_step(
        step_name="语义标准化与本体映射",
        input_data=input_data,
        output_content=output,
        context=context,
    )
    return ToolResponse(content=output)


async def run_step4_data_quality_repair(input_data: str, context: str = "") -> ToolResponse:
    data_dir = _get_default_step4_input_path(input_data)
    project_root = _get_project_root()
    workspace_dir = os.path.join(_get_program_output_root(), "step4_results")
    validation_config_path = os.path.join(project_root, "agent_4", "validation_config.json")
    print(f"[Step4] 使用输入目录: {data_dir}")

    if not os.path.isdir(data_dir):
        output = f"Step4: 输入目录不存在，无法执行数据质量检测与自动修复。输入目录: {data_dir}"
        _trace_collector.record_step(
            step_name="数据质量检测与自动修复",
            input_data=data_dir,
            output_content=output,
            context=context,
        )
        memory_feedback = await report_last_step_reflection(
            input_text=data_dir,
            duration_seconds=0.0,
            orchestrator_summary=output,
            retrieved_bullet_ids=[],
        )
        if memory_feedback:
            print(f"\n[Memory Supervisor Step4 反思结果]\n{memory_feedback}\n")
        return ToolResponse(content=output)

    started_at = time.time()
    session_messages: list[dict[str, Any]] = []
    session_tool_events: list[dict[str, Any]] = []
    try:
        prepared_input, prepared_json_count = _prepare_step4_input_dir(data_dir, workspace_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        agent4_module = importlib.import_module("agent_4.main")
        run_data_quality_repair = getattr(agent4_module, "run_data_quality_repair")

        result = await run_data_quality_repair(
            input_dir=prepared_input,
            workspace_dir=workspace_dir,
            validation_config_path=validation_config_path,
            verbose=True,
        )
        result["input_dir"] = data_dir
        result["prepared_input_csv"] = prepared_input if str(prepared_input).lower().endswith(".csv") else ""
        result["prepared_json_count"] = prepared_json_count
        output = _build_step4_summary(data_dir, result)
        session_messages, session_tool_events = _build_step4_trace_payload(result)
    except Exception as e:
        result = {
            "success": False,
            "input_dir": data_dir,
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
            "error": str(e),
        }
        output = (
            f"Step4: 数据质量检测与自动修复执行失败。\n"
            f"输入目录: {data_dir}\n错误: {e}"
        )
        session_messages, session_tool_events = _build_step4_trace_payload(result)

    _trace_collector.record_step(
        step_name="数据质量检测与自动修复",
        input_data=data_dir,
        output_content=output,
        context=context,
        raw_messages=session_messages,
        full_raw_messages=session_messages,
        tool_events=session_tool_events,
        full_tool_events=session_tool_events,
    )
    duration = time.time() - started_at
    memory_feedback = await report_last_step_reflection(
        input_text=data_dir,
        duration_seconds=duration,
        orchestrator_summary=output,
        retrieved_bullet_ids=[],
    )
    if memory_feedback:
        print(f"\n[Memory Supervisor Step4 反思结果]\n{memory_feedback}\n")
    return ToolResponse(content=output)


async def run_step5_task_oriented_clipping(input_data: str, context: str = "") -> ToolResponse:
    input_csv = _get_default_step5_input_csv()
    task_text = _get_default_step5_task_text()
    output_dir = _get_step5_results_dir()
    os.makedirs(output_dir, exist_ok=True)
    started_at = time.time()
    session_messages: list[dict[str, Any]] = []
    session_tool_events: list[dict[str, Any]] = []

    if not os.path.isfile(input_csv):
        result = {
            "success": False,
            "input_csv": input_csv,
            "task_text": task_text,
            "filtered_csv_path": "",
            "selection_report_path": "",
            "error": f"Step5 输入文件不存在: {input_csv}",
        }
        output = _build_step5_summary(input_csv, task_text, result)
        session_messages, session_tool_events = _build_step5_trace_payload(result)
        _trace_collector.record_step(
            step_name="任务导向裁剪",
            input_data=input_csv,
            output_content=output,
            context=context,
            raw_messages=session_messages,
            full_raw_messages=session_messages,
            tool_events=session_tool_events,
            full_tool_events=session_tool_events,
        )
        memory_feedback = await report_last_step_reflection(
            input_text=input_csv,
            duration_seconds=0.0,
            orchestrator_summary=output,
            retrieved_bullet_ids=[],
        )
        if memory_feedback:
            print(f"\n[Memory Supervisor Step5 反思结果]\n{memory_feedback}\n")
        return ToolResponse(content=output)

    try:
        if _get_project_root() not in sys.path:
            sys.path.insert(0, _get_project_root())
        selector_module = importlib.import_module("agent_5.medical_column_selector")
        selector_cls = getattr(selector_module, "TaskDrivenColumnSelector")
        selector = selector_cls()
        selection_result = await asyncio.to_thread(
            selector.run,
            input_csv,
            task_text,
            output_dir,
        )

        filtered_csv_path = str(selection_result.get("filtered_csv_path") or "")
        selection_report_path = str(selection_result.get("selection_report_path") or "")
        if not filtered_csv_path or not selection_report_path:
            raise RuntimeError("Step5 未返回完整输出路径。")
        if not os.path.isfile(filtered_csv_path):
            raise FileNotFoundError(f"Step5 筛选结果文件不存在: {filtered_csv_path}")
        if not os.path.isfile(selection_report_path):
            raise FileNotFoundError(f"Step5 选择报告文件不存在: {selection_report_path}")

        result = {
            "success": True,
            "input_csv": input_csv,
            "task_text": task_text,
            "filtered_csv_path": filtered_csv_path,
            "selection_report_path": selection_report_path,
            "error": "",
        }
        output = _build_step5_summary(input_csv, task_text, result)
        session_messages, session_tool_events = _build_step5_trace_payload(result)
    except Exception as e:
        result = {
            "success": False,
            "input_csv": input_csv,
            "task_text": task_text,
            "filtered_csv_path": "",
            "selection_report_path": "",
            "error": str(e),
        }
        output = _build_step5_summary(input_csv, task_text, result)
        session_messages, session_tool_events = _build_step5_trace_payload(result)

    _trace_collector.record_step(
        step_name="任务导向裁剪",
        input_data=input_csv,
        output_content=output,
        context=context,
        raw_messages=session_messages,
        full_raw_messages=session_messages,
        tool_events=session_tool_events,
        full_tool_events=session_tool_events,
    )
    duration = time.time() - started_at
    memory_feedback = await report_last_step_reflection(
        input_text=input_csv,
        duration_seconds=duration,
        orchestrator_summary=output,
        retrieved_bullet_ids=[],
    )
    if memory_feedback:
        print(f"\n[Memory Supervisor Step5 反思结果]\n{memory_feedback}\n")
    return ToolResponse(content=output)


async def run_step6_consistency_verification(input_data: str, context: str = "") -> ToolResponse:
    started_at = time.time()
    filtered_csv, selection_report = _resolve_latest_step5_artifacts()
    output_step6 = _get_step67_step6_output_dir()
    output_step7 = _get_step67_step7_output_dir()

    if not filtered_csv or not os.path.isfile(filtered_csv):
        result = {
            "success": False,
            "filtered_csv": filtered_csv,
            "selection_report": selection_report,
            "output_step6": output_step6,
            "output_step7": output_step7,
            "passed_patient_ids": [],
            "step6_report_txt": "",
            "step6_report_json": "",
            "error": f"未在 {_get_step5_results_dir()} 找到 Step5 输出 CSV（*_filtered_*.csv）",
        }
        output = _build_step6_summary(result)
        session_messages, session_tool_events = _build_step6_trace_payload(result)
        _trace_collector.record_step(
            step_name="一致性验证与置信度估计",
            input_data=filtered_csv or input_data,
            output_content=output,
            context=context,
            raw_messages=session_messages,
            full_raw_messages=session_messages,
            tool_events=session_tool_events,
            full_tool_events=session_tool_events,
        )
        memory_feedback = await report_last_step_reflection(
            input_text=filtered_csv or input_data,
            duration_seconds=0.0,
            orchestrator_summary=output,
            retrieved_bullet_ids=[],
        )
        if memory_feedback:
            print(f"\n[Memory Supervisor Step6 反思结果]\n{memory_feedback}\n")
        return ToolResponse(content=output)

    runtime_overrides = {
        "raw_csv": filtered_csv,
        "filtered_csv": filtered_csv,
        "selection_report": selection_report,
        "output_step6_dir": output_step6,
        "output_step7_dir": output_step7,
    }

    session_messages: list[dict[str, Any]] = []
    session_tool_events: list[dict[str, Any]] = []
    try:
        pipeline_module = _load_step67_pipeline_module()
        run_step6_only = getattr(pipeline_module, "run_step6_only")
        result = await run_step6_only(runtime_overrides=runtime_overrides)
        result["filtered_csv"] = filtered_csv
        result["selection_report"] = selection_report

        if not os.path.isfile(str(result.get("step6_report_txt") or "")):
            raise FileNotFoundError(f"Step6 文本报告不存在: {result.get('step6_report_txt', '')}")
        if not os.path.isfile(str(result.get("step6_report_json") or "")):
            raise FileNotFoundError(f"Step6 JSON 报告不存在: {result.get('step6_report_json', '')}")

        result["success"] = True
        result["error"] = ""
        _step67_runtime_cache.clear()
        _step67_runtime_cache.update({
            "runtime_overrides": runtime_overrides,
            "passed_patient_ids": result.get("passed_patient_ids") or [],
            "step6_result": result,
        })
        output = _build_step6_summary(result)
        session_messages, session_tool_events = _build_step6_trace_payload(result)
    except Exception as e:
        result = {
            "success": False,
            "filtered_csv": filtered_csv,
            "selection_report": selection_report,
            "output_step6": output_step6,
            "output_step7": output_step7,
            "passed_patient_ids": [],
            "step6_report_txt": "",
            "step6_report_json": "",
            "error": str(e),
        }
        output = _build_step6_summary(result)
        session_messages, session_tool_events = _build_step6_trace_payload(result)

    _trace_collector.record_step(
        step_name="一致性验证与置信度估计",
        input_data=filtered_csv,
        output_content=output,
        context=context,
        raw_messages=session_messages,
        full_raw_messages=session_messages,
        tool_events=session_tool_events,
        full_tool_events=session_tool_events,
    )
    duration = time.time() - started_at
    memory_feedback = await report_last_step_reflection(
        input_text=filtered_csv,
        duration_seconds=duration,
        orchestrator_summary=output,
        retrieved_bullet_ids=[],
    )
    if memory_feedback:
        print(f"\n[Memory Supervisor Step6 反思结果]\n{memory_feedback}\n")
    return ToolResponse(content=output)


async def run_step7_phenotype_knowledge_confirmation(input_data: str, context: str = "") -> ToolResponse:
    started_at = time.time()
    filtered_csv, selection_report = _resolve_latest_step5_artifacts()
    output_step6 = _get_step67_step6_output_dir()
    output_step7 = _get_step67_step7_output_dir()

    step6_rerun = False
    runtime_overrides = _step67_runtime_cache.get("runtime_overrides") or {
        "raw_csv": filtered_csv,
        "filtered_csv": filtered_csv,
        "selection_report": selection_report,
        "output_step6_dir": output_step6,
        "output_step7_dir": output_step7,
    }

    passed_patient_ids = list(_step67_runtime_cache.get("passed_patient_ids") or [])

    session_messages: list[dict[str, Any]] = []
    session_tool_events: list[dict[str, Any]] = []
    try:
        if not filtered_csv or not os.path.isfile(filtered_csv):
            raise FileNotFoundError(f"未在 {_get_step5_results_dir()} 找到 Step5 输出 CSV（*_filtered_*.csv）")
        if not selection_report or not os.path.isfile(selection_report):
            raise FileNotFoundError(f"未在 {_get_step5_results_dir()} 找到 Step5 输出报告（*_selection_report_*.json）")

        pipeline_module = _load_step67_pipeline_module()

        if not passed_patient_ids:
            step6_rerun = True
            run_step6_only = getattr(pipeline_module, "run_step6_only")
            step6_result = await run_step6_only(runtime_overrides=runtime_overrides)
            passed_patient_ids = list(step6_result.get("passed_patient_ids") or [])

        run_step7_only = getattr(pipeline_module, "run_step7_only")
        result = await run_step7_only(
            passed_patient_ids=passed_patient_ids,
            runtime_overrides=runtime_overrides,
        )
        result["filtered_csv"] = filtered_csv
        result["selection_report"] = selection_report

        if not os.path.isfile(str(result.get("step7_dataset_csv") or "")):
            raise FileNotFoundError(f"Step7 CSV 数据集不存在: {result.get('step7_dataset_csv', '')}")
        if not os.path.isfile(str(result.get("step7_dataset_jsonl") or "")):
            raise FileNotFoundError(f"Step7 JSONL 数据集不存在: {result.get('step7_dataset_jsonl', '')}")
        if not os.path.isfile(str(result.get("step7_dataset_json") or "")):
            raise FileNotFoundError(f"Step7 JSON 数据集不存在: {result.get('step7_dataset_json', '')}")

        result["success"] = True
        result["error"] = ""
        _step67_runtime_cache.clear()
        _step67_runtime_cache.update({
            "runtime_overrides": runtime_overrides,
            "passed_patient_ids": result.get("passed_patient_ids") or passed_patient_ids,
            "step7_result": result,
        })
        output = _build_step7_summary(result)
        session_messages, session_tool_events = _build_step7_trace_payload(result, step6_rerun=step6_rerun)
    except Exception as e:
        result = {
            "success": False,
            "filtered_csv": filtered_csv,
            "selection_report": selection_report,
            "output_step7": output_step7,
            "step7_dataset_csv": "",
            "step7_dataset_jsonl": "",
            "step7_dataset_json": "",
            "passed_patient_ids": passed_patient_ids,
            "error": str(e),
        }
        output = _build_step7_summary(result)
        session_messages, session_tool_events = _build_step7_trace_payload(result, step6_rerun=step6_rerun)

    _trace_collector.record_step(
        step_name="表型/知识确认",
        input_data=filtered_csv,
        output_content=output,
        context=context,
        raw_messages=session_messages,
        full_raw_messages=session_messages,
        tool_events=session_tool_events,
        full_tool_events=session_tool_events,
    )
    duration = time.time() - started_at
    memory_feedback = await report_last_step_reflection(
        input_text=filtered_csv,
        duration_seconds=duration,
        orchestrator_summary=output,
        retrieved_bullet_ids=[],
    )
    if memory_feedback:
        print(f"\n[Memory Supervisor Step7 反思结果]\n{memory_feedback}\n")
    return ToolResponse(content=output)
