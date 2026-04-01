import importlib
import json
import os
import time
from typing import Any

from agentscope.agent import ReActAgent
from agentscope.formatter import OllamaChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OllamaChatModel
from agentscope.tool import ToolResponse

from agent_1.agentscope_tool_loader import load_toolkit_from_config
from memory_agent.config import config
from memory_agent.memory_tool import get_playbook_context_data


_CONTEXT_PREVIEW_LIMIT = 300
_MESSAGE_TEXT_LIMIT = 1500
_TOOL_OUTPUT_LIMIT = 2000


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
        "output_root": os.path.join(project_root, "data"),
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
            thinking = str(block.get("thinking", ""))
            was_truncated = False
            if truncate:
                thinking, was_truncated = _truncate_text(block.get("thinking", ""), _MESSAGE_TEXT_LIMIT)
            item["thinking"] = thinking
            if was_truncated:
                truncated_flags.append(f"thinking_{idx}")
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
                    text_parts.append(f"[Thinking]\n{block.get('thinking', '')}\n[/Thinking]")
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

            chain = step.get("thinking_chain", [])
            if chain:
                parts.append(f"\n--- 完整推理链 ({len(chain)} 条消息) ---")
                for j, msg_item in enumerate(chain, 1):
                    role = msg_item.get("role", "?")
                    name = msg_item.get("name", "")
                    content = msg_item.get("content", "")
                    if not full and role == "system" and len(content) > 1000:
                        content = content[:1000] + "...(truncated)"
                    elif not full and role == "assistant" and len(content) > 3000:
                        content = content[:3000] + "...(truncated)"
                    parts.append(f"  [{j}] {name}({role}): {content}")
                parts.append("--- 推理链结束 ---\n")

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
        "5）最后输出最终报告。"
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
        formatter=OllamaChatFormatter(),
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
            {"type": "thinking", "thinking": fallback_note},
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
    output = "Step 4 (Mock): 检测完毕，数据质量良好。"
    _trace_collector.record_step(
        step_name="数据质量检测与自动修复",
        input_data=input_data,
        output_content=output,
        context=context,
    )
    return ToolResponse(content=output)


async def run_step5_task_oriented_clipping(input_data: str, context: str = "") -> ToolResponse:
    output = "Step 5 (Mock): 完成任务导向裁剪。"
    _trace_collector.record_step(
        step_name="任务导向裁剪",
        input_data=input_data,
        output_content=output,
        context=context,
    )
    return ToolResponse(content=output)


async def run_step6_consistency_verification(input_data: str, context: str = "") -> ToolResponse:
    output = "Step 6 (Mock): 一致性验证通过。"
    _trace_collector.record_step(
        step_name="一致性验证与置信度估计",
        input_data=input_data,
        output_content=output,
        context=context,
    )
    return ToolResponse(content=output)


async def run_step7_phenotype_knowledge_confirmation(input_data: str, context: str = "") -> ToolResponse:
    output = "Step 7 (Mock): 最终表型知识确认完成。"
    _trace_collector.record_step(
        step_name="表型/知识确认",
        input_data=input_data,
        output_content=output,
        context=context,
    )
    return ToolResponse(content=output)
