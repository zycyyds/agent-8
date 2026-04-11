import asyncio
import os
import sys
import time

import agentscope
from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import OllamaChatFormatter
from agentscope.message import Msg
from agentscope.model import OllamaChatModel
from agentscope.tool import Toolkit

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_orchestrator_dir = os.path.dirname(os.path.abspath(__file__))
if _orchestrator_dir not in sys.path:
    sys.path.insert(0, _orchestrator_dir)

from step_wrappers import (
    get_trace_collector,
    register_no_thinking_print_hook,
    run_step1_modal_recognition,
    run_step2_3_medical_data_cleaner,
    run_step2_parse_extract,
    run_step3_semantic_standardization,
    run_step4_data_quality_repair,
    run_step5_task_oriented_clipping,
    run_step6_consistency_verification,
    run_step7_phenotype_knowledge_confirmation,
)
from memory_supervisor import get_pipeline_context
from configs.loader import get_agent_config


AGENT_CFG = get_agent_config("main_orchestrator")


def _format_orchestrator_summary_content(content) -> str:
    if isinstance(content, list):
        text_parts = []
        fallback_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text = str(block.get("text", "")).strip()
                    if text:
                        text_parts.append(text)
                elif block_type == "tool_result":
                    output = str(block.get("output", "")).strip()
                    if output:
                        fallback_parts.append(output)
            else:
                fallback_parts.append(str(block).strip())

        combined = "\n".join(part for part in text_parts if part).strip()
        if combined:
            return combined
        return "\n".join(part for part in fallback_parts if part).strip()
    return str(content).strip()


def _build_orchestrator(context_str: str) -> ReActAgent:
    toolkit = Toolkit()
    toolkit.register_tool_function(run_step1_modal_recognition)
    toolkit.register_tool_function(run_step2_3_medical_data_cleaner)
    toolkit.register_tool_function(run_step2_parse_extract)
    toolkit.register_tool_function(run_step3_semantic_standardization)
    toolkit.register_tool_function(run_step4_data_quality_repair)
    toolkit.register_tool_function(run_step5_task_oriented_clipping)
    toolkit.register_tool_function(run_step6_consistency_verification)
    toolkit.register_tool_function(run_step7_phenotype_knowledge_confirmation)

    sys_prompt = f"""你是数据处理流水线的主协调者 (Orchestrator)。
你的任务是接收用户输入，并调用工具完成数据处理流水线。

【当前可用步骤】
1. 调用 `run_step1_modal_recognition`：数据感知与模态识别
2. 调用 `run_step2_3_medical_data_cleaner`：医学数据清洗、抽取、标准化与量纲统一
3. 调用 `run_step4_data_quality_repair`：数据质量检测与自动修复
4. 调用 `run_step5_task_oriented_clipping`：任务导向列裁剪

【执行规则】
1. 当用户输入路径时，先调用且仅调用一次 `run_step1_modal_recognition`。
2. 在 `run_step1_modal_recognition` 成功返回后，继续调用且仅调用一次 `run_step2_3_medical_data_cleaner`。
3. 在 `run_step2_3_medical_data_cleaner` 成功返回后，继续调用且仅调用一次 `run_step4_data_quality_repair`。
4. 在 `run_step4_data_quality_repair` 成功返回后，继续调用且仅调用一次 `run_step5_task_oriented_clipping`。
5. 如果工具支持 `context` 参数，把下方 ACE Playbook 原样传给工具。
6. `run_step2_3_medical_data_cleaner` 会进入交互式阶段；只有当用户在该阶段输入 `quit` 后，工具才会返回。
7. `run_step4_data_quality_repair` 默认会读取 `program/output/step2_3_results` 下最新的 `results_*` 目录作为输入。
8. `run_step5_task_oriented_clipping` 默认自动读取 `program/output/step4_results` 下最新的 `*_cleaned_*.csv` 作为输入。
9. 当你收到 `run_step5_task_oriented_clipping` 的有效结果后，再将整个流水线结果总结输出给用户，然后结束当前任务。
10. 你的最终总结、解释、追问、报错都必须使用中文输出；即使上游工具结果或原始医学文本是英文，也要用中文表述。
11. 不要调用未在【当前可用步骤】中列出的工具。
12. 只要工具已经返回了有效结果，就必须停止生成新的 JSON 工具调用。

【ACE Playbook (来源于你的 Memory Agent)】
{context_str}
"""

    model = OllamaChatModel(
        model_name=AGENT_CFG.get("model_name", "qwen3.5:4b"),
        options={
            "temperature": AGENT_CFG.get("temperature", 0.0),
            "seed": AGENT_CFG.get("seed", 666),
        },
    )

    agent = ReActAgent(
        name="MainOrchestrator",
        sys_prompt=sys_prompt,
        model=model,
        formatter=OllamaChatFormatter(),
        toolkit=toolkit,
        max_iters=15,
    )
    return register_no_thinking_print_hook(agent)


async def main():
    agentscope.init(project="MultiAgentPipeline", name="MainOrchestrator")

    user = UserAgent(name="User")

    msg = Msg(name="system", content="请输入待处理的图片或文档路径，启动流水线：", role="system")
    print("\nMainOrchestrator: 请输入待处理的图片或文档路径，启动流水线：")

    while True:
        try:
            try:
                msg = await user(msg)
            except KeyboardInterrupt:
                print("\nExiting...")
                break

            user_input = str(msg.content).strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break

            print("[Orchestrator] 正在按本轮输入检索 ACE Playbook...")
            context_str, used_bullet_ids = await get_pipeline_context(query_text=user_input)
            orchestrator = _build_orchestrator(context_str)

            start_time = time.time()

            trace_collector = get_trace_collector()
            trace_collector.clear()

            orchestrator_task = asyncio.create_task(orchestrator(msg))
            try:
                msg = await orchestrator_task
            except KeyboardInterrupt:
                await orchestrator.interrupt()
                msg = await orchestrator_task

            if not any(step.get("step_name") == "Step2_3 医学数据清洗与标准化" for step in trace_collector.steps):
                print("[Orchestrator] 未检测到 Step2_3，进行兜底补跑...")
                step2_result = await run_step2_3_medical_data_cleaner("", context_str)
                msg = Msg(name="system", content=step2_result.content, role="system")

            duration = time.time() - start_time

            print(f"\n[Orchestrator] 流水线执行耗时: {duration:.2f}s。")
            print(f"[Orchestrator] 共收集到 {len(trace_collector)} 个步骤的完整 Trace。")

            final_summary_text = _format_orchestrator_summary_content(msg.content)
            print(f"\n[Orchestrator] 最终输出:\n{final_summary_text}\n")

        except EOFError:
            break


if __name__ == "__main__":
    asyncio.run(main())
