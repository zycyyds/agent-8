from __future__ import annotations

import asyncio
import ast
import json
import re
from typing import Any, Callable

from agentscope.model import OllamaChatModel

from .models import (
    CuratorDelta,
    CuratorOperation,
    ReflectionResult,
    SECTION_ORDER,
    normalize_section,
)
from .playbook import ACEPlaybookManager, normalize_bullet_content
from .reflector import build_safe_fallback_rule
from .trace_parser import normalize_trace_payload


def _contains_cjk(text: Any) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


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


async def _translate_delta_to_chinese(model: Any, parsed: dict[str, Any]) -> dict[str, Any]:
    operations = parsed.get("operations", [])
    needs_translation = False

    if str(parsed.get("reasoning") or "").strip() and not _contains_cjk(parsed.get("reasoning")):
        needs_translation = True
    if not needs_translation:
        for item in operations:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if content and not _contains_cjk(content):
                needs_translation = True
                break

    if not needs_translation:
        return parsed

    translation_prompt = f"""把下面 JSON 中给人看的自然语言字符串翻译成中文，并保持 JSON 结构不变。

要求：
1. 只翻译 `reasoning` 和 `operations[].content` 这类自然语言字段。
2. 不要修改 `type`、`section` 等枚举值。
3. 工具名、路径、文件名可以保留原样。
4. 保持原意，不要扩写。
5. 只输出 JSON，不要附带解释。

原始 JSON：
{json.dumps(parsed, ensure_ascii=False, indent=2)}
"""

    response = model([{"role": "user", "content": translation_prompt}])
    if asyncio.iscoroutine(response):
        response = await response
    translated = _extract_json_object(
        _stringify_model_content(response.content if hasattr(response, "content") else response)
    )
    if not isinstance(translated, dict) or not translated:
        return parsed

    merged = dict(parsed)
    translated_reasoning = str(translated.get("reasoning") or "").strip()
    if translated_reasoning and _contains_cjk(translated_reasoning):
        merged["reasoning"] = translated_reasoning

    translated_operations = translated.get("operations", [])
    if isinstance(translated_operations, list):
        merged_ops: list[dict[str, Any]] = []
        for index, item in enumerate(operations):
            if not isinstance(item, dict):
                continue
            translated_item = translated_operations[index] if index < len(translated_operations) else {}
            merged_item = dict(item)
            translated_content = (
                str(translated_item.get("content") or "").strip()
                if isinstance(translated_item, dict)
                else ""
            )
            if translated_content and _contains_cjk(translated_content):
                merged_item["content"] = translated_content
            merged_ops.append(merged_item)
        merged["operations"] = merged_ops

    return merged


class ACECurator:
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

    async def propose_delta(
        self,
        manager: ACEPlaybookManager,
        reflection: ReflectionResult,
        trace_json: str | dict[str, Any],
    ) -> CuratorDelta:
        trace_data = normalize_trace_payload(trace_json)
        model = self._build_model()

        input_text = str(trace_data.get("input_text") or "")
        output_text = str(trace_data.get("orchestrator_summary") or "")
        current_playbook, _, _ = manager.format_playbook(query_text=input_text, max_bullets=30)
        reflection_text = json.dumps(reflection.to_dict(), ensure_ascii=False, indent=2)
        sections_text = ", ".join(SECTION_ORDER)
        prompt = f"""你是 ACE 框架中的 Curator。你的职责是把最近一次 reflection 转成增量 playbook 更新。
你不能重写整个 playbook，只能提出缺失的新条目。

要求：
1. 只输出新的增量操作，不要重复已有条目。
2. 只允许 `ADD` 操作。
3. `section` 只能从以下集合中选择：{sections_text}
4. `content` 必须使用中文，且要短小、可执行、适合长期复用，不能是总结、统计或路径清单。
5. 输出必须是 JSON，不要附带 markdown。

任务上下文：
输入: {input_text}
输出摘要: {output_text}

Recent Reflection:
{reflection_text}

Current Playbook Stats:
{manager.get_stats_text()}

Current Playbook:
{current_playbook}

输出格式：
{{
  "reasoning": "为什么需要这些增量",
  "operations": [
    {{
      "type": "ADD",
      "section": "validation_checklist",
      "content": "新增经验"
    }}
  ]
}}
"""

        response = model([{"role": "user", "content": prompt}])
        if asyncio.iscoroutine(response):
            response = await response

        parsed = _extract_json_object(
            _stringify_model_content(response.content if hasattr(response, "content") else response)
        )
        parsed = await _translate_delta_to_chinese(model, parsed)

        operations = []
        for item in parsed.get("operations", []):
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").upper() != "ADD":
                continue
            content = normalize_bullet_content(item.get("content"))
            if not content:
                continue
            section = normalize_section(item.get("section"), content)
            operations.append(CuratorOperation(type="ADD", section=section, content=content))

        if not operations:
            fallback_content = normalize_bullet_content(reflection.key_insight)
            if fallback_content:
                operations = [
                    CuratorOperation(
                        type="ADD",
                        section=normalize_section(reflection.key_insight_section, fallback_content),
                        content=fallback_content,
                    )
                ]

        if not operations:
            fallback_section, fallback_content = build_safe_fallback_rule(trace_data)
            operations = [
                CuratorOperation(
                    type="ADD",
                    section=fallback_section,
                    content=fallback_content,
                )
            ]

        return CuratorDelta(
            reasoning=str(parsed.get("reasoning", "")).strip() or "基于已验证 reflection 生成增量更新。",
            operations=operations,
        )

    def apply_delta(
        self,
        manager: ACEPlaybookManager,
        delta: CuratorDelta,
        source: str = "curator",
    ) -> dict[str, Any]:
        created_ids = []
        skipped_duplicates = 0
        applied_operations = []

        for op in delta.operations:
            if op.type != "ADD":
                continue
            result = manager.add_bullet(
                content=op.content,
                section=op.section,
                metadata={"source": source},
            )
            applied_operations.append(op.to_dict())
            if result.get("created"):
                created_ids.append(result["bullet_id"])
            elif result.get("reason") == "duplicate":
                skipped_duplicates += 1

        return {
            "operation": "curator_apply_delta",
            "reasoning": delta.reasoning,
            "applied_operations": applied_operations,
            "new_bullet_ids": created_ids,
            "new_bullets": len(created_ids),
            "skipped_duplicates": skipped_duplicates,
        }


MemoryCurator = ACECurator
