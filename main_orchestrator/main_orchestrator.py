import argparse
import asyncio
import functools
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main data pipeline orchestrator.")
    parser.add_argument(
        "--disable-memory-agent",
        action="store_true",
        help="Disable memory reflection and memory writes for the current pipeline run.",
    )
    return parser.parse_args(argv)


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


def _build_orchestrator(context_str: str, enable_memory_agent: bool = True) -> ReActAgent:
    _orchestrator_enable_memory_agent = enable_memory_agent

    def _bind_memory_toggle(step_fn):
        @functools.wraps(step_fn)
        async def _tool(input_data: str, context: str = "", enable_memory_agent: bool | None = None):
            _ = enable_memory_agent
            return await step_fn(
                input_data,
                context,
                enable_memory_agent=_orchestrator_enable_memory_agent,
            )

        return _tool

    toolkit = Toolkit()
    toolkit.register_tool_function(_bind_memory_toggle(run_step1_modal_recognition))
    toolkit.register_tool_function(_bind_memory_toggle(run_step2_3_medical_data_cleaner))
    toolkit.register_tool_function(run_step2_parse_extract)
    toolkit.register_tool_function(run_step3_semantic_standardization)
    toolkit.register_tool_function(_bind_memory_toggle(run_step4_data_quality_repair))
    toolkit.register_tool_function(_bind_memory_toggle(run_step5_task_oriented_clipping))
    toolkit.register_tool_function(_bind_memory_toggle(run_step6_consistency_verification))
    toolkit.register_tool_function(_bind_memory_toggle(run_step7_phenotype_knowledge_confirmation))

    sys_prompt = f"""你是数据处理流水线的主协调者 (Orchestrator)。
你的任务是接收用户输入，并调用工具完成数据处理流水线。

【当前可用步骤】
1. 调用 `run_step1_modal_recognition`：数据感知与模态识别
2. 调用 `run_step2_3_medical_data_cleaner`：医学数据清洗、抽取、标准化与量纲统一
3. 调用 `run_step4_data_quality_repair`：数据质量检测与自动修复
4. 调用 `run_step5_task_oriented_clipping`：任务导向列裁剪
5. 调用 `run_step6_consistency_verification`：一致性验证与置信度估计
6. 调用 `run_step7_phenotype_knowledge_confirmation`：表型/知识确认并生成训练数据集

【执行规则】
1. 当用户输入路径时，先调用且仅调用一次 `run_step1_modal_recognition`。
2. 在 `run_step1_modal_recognition` 成功返回后，继续调用且仅调用一次 `run_step2_3_medical_data_cleaner`。
3. 在 `run_step2_3_medical_data_cleaner` 成功返回后，继续调用且仅调用一次 `run_step4_data_quality_repair`。
4. 在 `run_step4_data_quality_repair` 成功返回后，继续调用且仅调用一次 `run_step5_task_oriented_clipping`。
5. 在 `run_step5_task_oriented_clipping` 成功返回后，继续调用且仅调用一次 `run_step6_consistency_verification`。
6. 在 `run_step6_consistency_verification` 成功返回后，继续调用且仅调用一次 `run_step7_phenotype_knowledge_confirmation`。
7. 如果工具支持 `context` 参数，把下方 ACE Playbook 原样传给工具。
8. `run_step2_3_medical_data_cleaner` 会自动完成一次性处理并直接返回，不会进入交互式 `quit` 阶段。
9. `run_step4_data_quality_repair` 优先读取 `program/output/step2_3_results/next_input/input.csv`，若不存在再回退到旧扫描逻辑。
10. `run_step5_task_oriented_clipping` 输出固定写入 `program/output/step5_results`。
11. `run_step6_consistency_verification` 与 `run_step7_phenotype_knowledge_confirmation` 的输入来自 `program/output/step5_results` 内最新 `*_filtered_*.csv`，输出写入 `program/output/step6-7_results`。
12. 当你收到 `run_step7_phenotype_knowledge_confirmation` 的有效结果后，再将整个流水线结果总结输出给用户，然后结束当前任务。
13. 你的最终总结、解释、追问、报错都必须使用中文输出；即使上游工具结果或原始医学文本是英文，也要用中文表述。
14. 不要调用未在【当前可用步骤】中列出的工具。
15. 只要工具已经返回了有效结果，就必须停止生成新的 JSON 工具调用。

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


async def main(enable_memory_agent: bool = True):
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

            if enable_memory_agent:
                print("[Orchestrator] 本次运行启用 memory_agent。")
            else:
                print("[Orchestrator] 本次运行已关闭 memory_agent。")

            print("[Orchestrator] 正在按本轮输入检索 ACE Playbook...")
            context_str, used_bullet_ids = await get_pipeline_context(query_text=user_input)
            orchestrator = _build_orchestrator(context_str, enable_memory_agent=enable_memory_agent)

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
                step2_result = await run_step2_3_medical_data_cleaner(
                    "",
                    context_str,
                    enable_memory_agent=enable_memory_agent,
                )
                msg = Msg(name="system", content=step2_result.content, role="system")

            duration = time.time() - start_time

            print(f"\n[Orchestrator] 流水线执行耗时: {duration:.2f}s。")
            print(f"[Orchestrator] 共收集到 {len(trace_collector)} 个步骤的完整 Trace。")

            final_summary_text = _format_orchestrator_summary_content(msg.content)
            print(f"\n[Orchestrator] 最终输出:\n{final_summary_text}\n")

        except EOFError:
            break


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(enable_memory_agent=not args.disable_memory_agent))
