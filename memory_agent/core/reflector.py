from __future__ import annotations

import asyncio
import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agentscope.model import OllamaChatModel
from pydantic import BaseModel, Field

from .models import Bullet, BulletTag, ReflectionResult, normalize_section
from .trace_parser import normalize_trace_payload


REFLECTOR_DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug" / "reflector"
_TRANSLATABLE_REFLECTION_FIELDS = (
    "analysis",
    "reasoning",
    "error_identification",
    "root_cause_analysis",
    "correct_approach",
    "key_insight",
)


def _stringify_model_content(content: Any) -> str:
    if isinstance(content, list):
        text_parts: list[str] = []
        fallback_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block_type == "thinking":
                    continue
                elif block_type == "tool_result":
                    fallback_parts.append(str(block.get("output", "")))
                else:
                    fallback_parts.append(str(block))
            else:
                fallback_parts.append(str(block))
        if any(part.strip() for part in text_parts):
            return "\n".join(part for part in text_parts if part).strip()
        return "\n".join(part for part in fallback_parts if part).strip()
    return str(content or "").strip()


def _iter_json_candidates(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    candidates: list[str] = [raw]

    match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        candidates.append(match.group(1).strip())
    else:
        match = re.search(r"```(?:\w+)?\s*(.*?)\s*```", raw, re.DOTALL)
        if match:
            candidates.append(match.group(1).strip())

    brace_match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1).strip())

    normalized_quotes = (
        raw.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    if normalized_quotes != raw:
        candidates.append(normalized_quotes)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        deduped.append(stripped)
        trimmed_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", stripped)
        if trimmed_trailing_commas not in seen:
            seen.add(trimmed_trailing_commas)
            deduped.append(trimmed_trailing_commas)
    return deduped


def _extract_json_object(text: str) -> dict[str, Any]:
    for candidate in _iter_json_candidates(text):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _contains_cjk(text: Any) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


async def _translate_reflection_strings_to_chinese(model: Any, parsed: dict[str, Any]) -> dict[str, Any]:
    fields_to_translate = {
        field: str(parsed.get(field) or "").strip()
        for field in _TRANSLATABLE_REFLECTION_FIELDS
        if str(parsed.get(field) or "").strip() and not _contains_cjk(parsed.get(field))
    }
    if not fields_to_translate:
        return parsed

    translation_prompt = f"""把下面 JSON 中需要给人看的自然语言字符串翻译成中文，并保持 JSON 结构稳定。

要求：
1. 只翻译自然语言字段，不要修改 JSON key。
2. `key_insight_section` 必须保持原枚举值，不要翻译。
3. `bullet_tags` 里的 `id` 和 `tag` 不要修改。
4. 工具名、路径、文件名、JSON key 可以保留原样。
5. 保持原意，不要扩写，不要引入新事实。
6. 只输出 JSON，不要附带解释。

优先翻译这些字段：
{json.dumps(fields_to_translate, ensure_ascii=False, indent=2)}

原始 JSON：
{json.dumps(parsed, ensure_ascii=False, indent=2)}
"""

    response = model([{"role": "user", "content": translation_prompt}])
    if asyncio.iscoroutine(response):
        response = await response
    translated, _, _ = _extract_reflector_payload(response)
    if not isinstance(translated, dict) or not translated:
        return parsed

    merged = dict(parsed)
    for field in _TRANSLATABLE_REFLECTION_FIELDS:
        translated_value = str(translated.get(field) or "").strip()
        if translated_value and _contains_cjk(translated_value):
            merged[field] = translated_value
    return merged


class _ReflectorBulletTagModel(BaseModel):
    id: str = ""
    tag: str = "neutral"


class _ReflectorStructuredOutputModel(BaseModel):
    analysis: str = ""
    key_insight: str = ""
    key_insight_section: str = ""
    bullet_tags: list[_ReflectorBulletTagModel] = Field(default_factory=list)


def _iter_tool_events(trace_data: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for step in trace_data.get("steps", []):
        for event in step.get("tool_events", []):
            if isinstance(event, dict):
                events.append(event)
    return events


def _has_mixed_modality_validation(trace_data: dict[str, Any]) -> bool:
    events = _iter_tool_events(trace_data)
    if any(event.get("tool_name") == "prepare_and_launch_validation_gui" for event in events):
        return True

    joined = json.dumps(trace_data, ensure_ascii=False)
    return "ocr+figure" in joined and "验证" in joined


def _has_tool_call_issue(trace_data: dict[str, Any]) -> bool:
    for event in _iter_tool_events(trace_data):
        status = str(event.get("status") or "")
        error = str(event.get("error") or "")
        if status == "failed":
            return True
        if any(keyword in error for keyword in ("unexpected keyword", "Traceback", "Error:", "失败", "异常")):
            return True
    return False


def _has_summary_mismatch(trace_data: dict[str, Any]) -> bool:
    summary = str(trace_data.get("orchestrator_summary") or "")
    if not summary:
        return False
    if any(token in summary for token in ("准确率", "/Users/", "output/", "data/", ".json")):
        return True

    outputs = "\n".join(str(event.get("tool_output") or "") for event in _iter_tool_events(trace_data))
    for token in re.findall(r"(?:[A-Za-z0-9_./-]+(?:\.json|/))", summary):
        if token and token not in outputs:
            return True
    return False


def build_safe_fallback_rule(trace_data: dict[str, Any]) -> tuple[str, str]:
    if _has_mixed_modality_validation(trace_data):
        return (
            "validation_checklist",
            "检测到混合模态样本时，必须先完成人工验证并回写验证结果，再进入数据整理。",
        )
    if _has_tool_call_issue(trace_data):
        return (
            "tool_usage",
            "调用工具时必须严格遵守参数签名，并在每次工具返回后核对状态再继续后续步骤。",
        )
    if _has_summary_mismatch(trace_data):
        return (
            "output_contracts",
            "对用户的结论文本必须严格依据工具返回结果生成，不得自行补充路径、准确率或统计字段。",
        )
    return (
        "failure_patterns",
        "反思结构化失败时，必须只依据结构化 trace 和工具结果生成规则，不得直接复用自然语言总结。",
    )


def _build_reflector_primary_trace(trace_data: dict[str, Any]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for step in trace_data.get("steps", []):
        if not isinstance(step, dict):
            continue
        assistant_outputs = [
            str(item).strip()
            for item in step.get("assistant_outputs", [])
            if str(item).strip()
        ]
        tool_events = [
            dict(event)
            for event in step.get("tool_events", [])
            if isinstance(event, dict)
        ]
        steps.append(
            {
                "step_name": str(step.get("step_name") or ""),
                "timestamp": str(step.get("timestamp") or ""),
                "input_data": str(step.get("input_data") or ""),
                "context_preview": str(step.get("context_preview") or ""),
                "assistant_outputs": assistant_outputs,
                "tool_events": tool_events,
                "final_output": str(step.get("final_output") or ""),
            }
        )

    return {
        "input_text": str(trace_data.get("input_text") or ""),
        "duration_seconds": float(trace_data.get("duration_seconds") or 0.0),
        "orchestrator_summary": str(trace_data.get("orchestrator_summary") or ""),
        "retrieved_bullet_ids": [
            str(item)
            for item in trace_data.get("retrieved_bullet_ids", [])
            if str(item).strip()
        ],
        "steps": steps,
        "success": bool(trace_data.get("success", True)),
        "raw_trace_text": str(trace_data.get("raw_trace_text") or ""),
        "full_raw_trace_text": str(trace_data.get("full_raw_trace_text") or trace_data.get("raw_trace_text") or ""),
        "failure_signals": [
            str(item)
            for item in trace_data.get("failure_signals", [])
            if str(item).strip()
        ],
    }


def _preview_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _build_trace_evidence_summary(trace_data: dict[str, Any]) -> str:
    lines = [
        f"输入: {trace_data.get('input_text', '')}",
        f"耗时: {trace_data.get('duration_seconds', 0.0)}s",
        f"检索到的 bullet IDs: {trace_data.get('retrieved_bullet_ids', [])}",
        f"执行状态: {'成功' if bool(trace_data.get('success', True)) else '失败'}",
    ]

    orchestrator_summary = str(trace_data.get("orchestrator_summary") or "").strip()
    if orchestrator_summary:
        lines.append(f"编排器总结(仅供参考): {_preview_text(orchestrator_summary, 180)}")

    for idx, step in enumerate(trace_data.get("steps", []), 1):
        tool_names = [
            str(event.get("tool_name") or "")
            for event in step.get("tool_events", [])
            if str(event.get("tool_name") or "")
        ]
        failed = [
            str(event.get("tool_name") or "")
            for event in step.get("tool_events", [])
            if str(event.get("status") or "") == "failed"
        ]
        lines.append(
            f"Step {idx}: {step.get('step_name', '')} | tools={tool_names or ['无']} | failed={failed or ['无']}"
        )
        assistant_outputs = [
            str(item).strip()
            for item in step.get("assistant_outputs", [])
            if str(item).strip()
        ]
        if assistant_outputs:
            lines.append(f"Step {idx} assistant 输出: {_preview_text(assistant_outputs[0])}")
        final_output = str(step.get("final_output") or "").strip()
        if final_output:
            lines.append(f"Step {idx} 最终输出: {_preview_text(final_output, 260)}")

    raw_trace_text = str(trace_data.get("raw_trace_text") or "").strip()
    if raw_trace_text and not trace_data.get("steps"):
        lines.append(f"Legacy Trace 摘要: {raw_trace_text[:500]}")

    return "\n".join(lines).strip()


def _collect_trace_tool_names(trace_data: dict[str, Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for event in _iter_tool_events(trace_data):
        name = str(event.get("tool_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _extract_reflector_payload(response: Any) -> tuple[dict[str, Any], str, str]:
    metadata = getattr(response, "metadata", None)
    if isinstance(metadata, dict) and metadata:
        return dict(metadata), json.dumps(metadata, ensure_ascii=False, indent=2), "metadata"

    model_text = _stringify_model_content(response.content if hasattr(response, "content") else response)
    return _extract_json_object(model_text), model_text, "content"


def _default_reasoning(parsed: dict[str, Any], key_insight: str) -> str:
    return (
        str(parsed.get("reasoning") or parsed.get("analysis") or "").strip()
        or f"已依据结构化 trace 提炼一条可复用规则：{key_insight}"
    )


def _default_error_identification(success: bool) -> str:
    if success:
        return "本轮未发现需要单独记录的新增错误，主要目标是沉淀可复用规则。"
    return "本轮存在需要沉淀为规则的执行偏差。"


def _default_root_cause_analysis() -> str:
    return "执行过程中的关键判断仍依赖即时生成，缺少稳定规则约束时容易重复出现相同决策偏差。"


def _default_correct_approach(key_insight: str) -> str:
    return key_insight


def _save_reflector_debug_artifact(
    *,
    trace_data: dict[str, Any],
    used_bullets: list[Bullet],
    evidence_summary: str,
    full_input_json: str,
    playbook_text: str,
    prompt: str,
    raw_reflector_output: str,
    output_source: str,
    parsed_payload: dict[str, Any],
    repair_attempts: list[dict[str, Any]],
    final_reflection: ReflectionResult,
    fallback_applied: bool,
    fallback_reason: str,
) -> dict[str, str]:
    reflector_input = {
        "trace_data": trace_data,
        "used_bullets": [bullet.to_dict() for bullet in used_bullets],
        "evidence_summary": evidence_summary,
        "full_input_json": full_input_json,
        "playbook_text": playbook_text,
        "prompt": prompt,
    }
    debug_payload = {
        "saved_at": datetime.now().isoformat(),
        "reflector_input": reflector_input,
        "reflector_output": {
            "output_source": output_source,
            "fallback_applied": fallback_applied,
            "fallback_reason": fallback_reason,
            "raw_reflector_output": raw_reflector_output,
            "parsed_payload": parsed_payload,
            "repair_attempts": repair_attempts,
            "final_reflection": final_reflection.to_dict(),
        },
    }

    REFLECTOR_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    history_path = REFLECTOR_DEBUG_DIR / f"reflector_response_{timestamp}.json"
    last_path = REFLECTOR_DEBUG_DIR / "last_reflector_response.json"
    input_history_path = REFLECTOR_DEBUG_DIR / f"reflector_input_{timestamp}.json"
    input_last_path = REFLECTOR_DEBUG_DIR / "last_reflector_input.json"
    serialized = json.dumps(debug_payload, ensure_ascii=False, indent=2)
    input_serialized = json.dumps(reflector_input, ensure_ascii=False, indent=2)
    history_path.write_text(serialized, encoding="utf-8")
    last_path.write_text(serialized, encoding="utf-8")
    input_history_path.write_text(input_serialized, encoding="utf-8")
    input_last_path.write_text(input_serialized, encoding="utf-8")
    return {
        "debug_input_file": str(input_last_path),
        "debug_input_history_file": str(input_history_path),
        "debug_file": str(last_path),
        "debug_history_file": str(history_path),
    }


class ACEReflector:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        seed: int,
        model_cls=OllamaChatModel,
        model_factory: Callable[[], Any] | None = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.model_cls = model_cls
        self.model_factory = model_factory
        self.last_debug_artifact: dict[str, str] | None = None

    def _build_model(self):
        if self.model_factory is not None:
            return self.model_factory()
        return self.model_cls(
            model_name=self.model_name,
            options={
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": 4096,
            },
        )

    async def analyze(
        self,
        trace_json: str | dict[str, Any],
        used_bullets: list[Bullet],
        max_refinement_rounds: int = 3,
    ) -> ReflectionResult:
        trace_data = _build_reflector_primary_trace(normalize_trace_payload(trace_json))
        success = bool(trace_data.get("success", True)) and not _has_tool_call_issue(trace_data)

        model = self._build_model()

        playbook_lines = [f"[{bullet.id}] {bullet.content}" for bullet in used_bullets]
        playbook_text = "\n".join(playbook_lines) or "无"
        evidence_summary = _build_trace_evidence_summary(trace_data)
        trace_tool_names = _collect_trace_tool_names(trace_data)
        trace_tools_text = "、".join(trace_tool_names) if trace_tool_names else "无"
        full_input_json = json.dumps(trace_data, ensure_ascii=False, indent=2)
        repair_attempts: list[dict[str, Any]] = []

        prompt = f"""你是 ACE 框架中的 Reflector，负责诊断一次执行轨迹为什么成功或失败。
你只做反思，不负责直接改写 playbook。
输入已经是预结构化 trace，请严格基于证据输出 JSON。

要求：
1. 只基于证据判断 bullet 是 helpful / harmful / neutral。
2. `key_insight` 必须是单条、命令式、可长期复用的规则。
3. 所有字符串字段必须使用中文，不要输出英文规则，不要举与当前任务无关的通用 JSON 示例。
4. 优先输出“工具使用策略”，也就是工具调用顺序、工具输入约束、工具输出校验、调用后下一步动作。
5. 只有当证据明显表明这是业务规则或验证规则时，才优先选择 `validation_checklist`、`modality_rules`、`data_organization` 等其他 section。
6. 如果工具链已经很清楚，`key_insight_section` 优先填写 `tool_usage`。
7. 为了提高稳定性，你只需要输出最小必需字段，不要补充多余字段。
8. 只输出 JSON，不要附带 markdown。
9. 结构化轨迹摘要只用于导航，不是主证据；你的主要依据必须是完整 step trace JSON。
10. 如摘要与完整 step trace 冲突，以完整 step trace 为准。
11. 请优先依据每个 step 的 `assistant_outputs`、`tool_events`、`final_output` 提炼 insight，不要把 `thinking` 当作主要证据。

结构化轨迹摘要（仅用于导航，不是主证据）：
{evidence_summary}

本次 trace 中出现的工具：
{trace_tools_text}

完整 step trace（主证据，包含前序 step 的 assistant 可见输出、工具使用与工具返回）：
{full_input_json}

Generator 使用过的 playbook 条目：
{playbook_text}

请输出最小 JSON：
{{
  "analysis": "可选，一句简短说明；如果拿不准可以留空",
  "key_insight": "应被长期记住的一条经验",
  "key_insight_section": "validation_checklist|tool_usage|output_contracts|failure_patterns|strategies_and_hard_rules|modality_rules|data_organization|domain_heuristics",
  "bullet_tags": [
    {{"id": "bullet_id", "tag": "helpful|harmful|neutral"}}
  ]
}}
"""

        try:
            response = model(
                [{"role": "user", "content": prompt}],
                structured_model=_ReflectorStructuredOutputModel,
            )
        except TypeError:
            response = model([{"role": "user", "content": prompt}])
        if asyncio.iscoroutine(response):
            response = await response

        parsed, raw_reflector_output, output_source = _extract_reflector_payload(response)
        parsed = await _translate_reflection_strings_to_chinese(model, parsed)

        repair_rounds = max(0, int(max_refinement_rounds or 1) - 1)
        while not parsed and repair_rounds > 0:
            repair_prompt = f"""把下面内容修正为一个合法 JSON 对象，只输出 JSON，不要附带解释。

必须包含这些字段：
- analysis: string
- key_insight: string
- key_insight_section: string
- bullet_tags: array

如果某个字段缺失，请补空字符串或空数组。
原始内容：
{raw_reflector_output}
"""
            try:
                repair_response = model(
                    [{"role": "user", "content": repair_prompt}],
                    structured_model=_ReflectorStructuredOutputModel,
                )
            except TypeError:
                repair_response = model([{"role": "user", "content": repair_prompt}])
            if asyncio.iscoroutine(repair_response):
                repair_response = await repair_response
            parsed, raw_reflector_output, output_source = _extract_reflector_payload(repair_response)
            parsed = await _translate_reflection_strings_to_chinese(model, parsed)
            repair_attempts.append(
                {
                    "round": len(repair_attempts) + 1,
                    "output_source": output_source,
                    "raw_output": raw_reflector_output,
                    "parsed_payload": parsed,
                }
            )
            repair_rounds -= 1

        bullet_tags: list[BulletTag] = []
        for item in parsed.get("bullet_tags", []):
            if not isinstance(item, dict):
                continue
            bullet_id = str(item.get("id") or "").strip()
            tag = str(item.get("tag") or "neutral").strip().lower()
            if tag not in {"helpful", "harmful", "neutral"} or not bullet_id:
                continue
            bullet_tags.append(BulletTag(id=bullet_id, tag=tag))
        if not bullet_tags:
            for bullet_id in parsed.get("helpful_bullet_ids", []):
                if str(bullet_id).strip():
                    bullet_tags.append(BulletTag(id=str(bullet_id).strip(), tag="helpful"))
            for bullet_id in parsed.get("harmful_bullet_ids", []):
                if str(bullet_id).strip():
                    bullet_tags.append(BulletTag(id=str(bullet_id).strip(), tag="harmful"))

        key_insight = str(parsed.get("key_insight") or "").strip()
        key_insight_section = normalize_section(parsed.get("key_insight_section"), key_insight)

        if not parsed or not key_insight:
            fallback_section, fallback_content = build_safe_fallback_rule(trace_data)
            result = ReflectionResult(
                reasoning=str(parsed.get("reasoning") or "").strip() or "结构化输出不完整，已使用确定性回退规则。",
                error_identification=(
                    str(parsed.get("error_identification") or "").strip()
                    or ("模型未返回可解析 JSON。" if not parsed else "`key_insight` 为空，无法直接沉淀规则。")
                ),
                root_cause_analysis=(
                    str(parsed.get("root_cause_analysis") or "").strip()
                    or "Reflector 输出格式不稳定，或缺少可直接沉淀的核心 insight。"
                ),
                correct_approach=(
                    str(parsed.get("correct_approach") or "").strip()
                    or "应仅基于结构化 trace 与工具结果提炼单条规则；当结构化结果不完整时再回退。"
                ),
                key_insight=fallback_content,
                key_insight_section=fallback_section,
                bullet_tags=bullet_tags,
                is_success=success,
            )
            fallback_reason = "unparseable_json" if not parsed else "missing_key_insight"
            self.last_debug_artifact = _save_reflector_debug_artifact(
                trace_data=trace_data,
                used_bullets=used_bullets,
                evidence_summary=evidence_summary,
                full_input_json=full_input_json,
                playbook_text=playbook_text,
                prompt=prompt,
                raw_reflector_output=raw_reflector_output,
                output_source=output_source,
                parsed_payload=parsed,
                repair_attempts=repair_attempts,
                final_reflection=result,
                fallback_applied=True,
                fallback_reason=fallback_reason,
            )
            return result

        result = ReflectionResult(
            reasoning=_default_reasoning(parsed, key_insight),
            error_identification=str(parsed.get("error_identification") or "").strip() or _default_error_identification(success),
            root_cause_analysis=str(parsed.get("root_cause_analysis") or "").strip() or _default_root_cause_analysis(),
            correct_approach=str(parsed.get("correct_approach") or "").strip() or _default_correct_approach(key_insight),
            key_insight=key_insight,
            key_insight_section=key_insight_section,
            bullet_tags=bullet_tags,
            is_success=success,
        )
        self.last_debug_artifact = _save_reflector_debug_artifact(
            trace_data=trace_data,
            used_bullets=used_bullets,
            evidence_summary=evidence_summary,
            full_input_json=full_input_json,
            playbook_text=playbook_text,
            prompt=prompt,
            raw_reflector_output=raw_reflector_output,
            output_source=output_source,
            parsed_payload=parsed,
            repair_attempts=repair_attempts,
            final_reflection=result,
            fallback_applied=False,
            fallback_reason="",
        )
        return result


StrategyReflector = ACEReflector
