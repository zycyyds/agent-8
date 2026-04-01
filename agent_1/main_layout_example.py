import asyncio
import os
import sys

import agentscope
from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import OllamaChatFormatter
from agentscope.message import Msg
from agentscope.model import OllamaChatModel
from agentscope.memory import InMemoryMemory, Mem0LongTermMemory
from agentscope.embedding import OllamaTextEmbedding

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentscope_tool_loader import load_toolkit_from_config
import layout_analysis_tool

# 导入配置加载器
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from configs.loader import get_agent_config

# 获取本智能体的模型配置
AGENT_CFG = get_agent_config("modal_identification")

# memory tools removed


async def main():
    agentscope.init(project="LayoutAnalysisAgent", name="Modal identification")

    toolkit = load_toolkit_from_config(os.path.join(os.path.dirname(__file__), "agentscope_tools.json"))

    # Memory 工具已经移交给独立的主协调者使用


    model = OllamaChatModel(
        model_name=AGENT_CFG.get("model_name", "qwen3.5:9b"),
        options={
            "temperature": AGENT_CFG.get("temperature", 0.0),
            "seed": AGENT_CFG.get("seed", 666),
        },
    )
    formatter = OllamaChatFormatter()

    # 初始化 Embedding Model (用于长期记忆的向量检索)
    embedding_model = OllamaTextEmbedding(
        model_name=AGENT_CFG.get("embedding_model", "nomic-embed-text"),
        dimensions=AGENT_CFG.get("dimensions", 768)
    )

    # 初始化基于 Mem0 的长时记忆引擎
    long_term_memory = Mem0LongTermMemory(
        agent_name="Modal identification",
        user_name="User",
        model=model,
        embedding_model=embedding_model,
        on_disk=False
    )

    context_str = ""

    # 动态构建系统提示词
    base_sys_prompt = (
        "你是一个布局分析助手。你只能使用工具完成图片布局分析、人工验证和数据集整理。并且保持和用户用中文交流。\n"
        "用户只会提供图片文件路径或目录路径，或请求对已有结果进行人工验证。\n"
        "【执行步骤】请严格按照以下步骤并**必须只使用标准的JSON格式调用所有工具**，绝对禁止使用 `<tool_call>` 或 HTML 标签！\n"
        "对于不需要参数的工具，例如 prepare_and_launch_validation_gui，你应该输出类似以下纯 JSON 格式：\n"
        "{\n"
        "    \"type\": \"tool_use\",\n"
        "    \"name\": \"prepare_and_launch_validation_gui\",\n"
        "    \"input\": {}\n"
        "}\n\n"
        "1）调用名为 `Collect Image Files` 的工具，传入用户原始输入作为 `target_path`。该工具会自动分拣文本/表格文件到 data 目录，并返回**待处理的图片列表**及**图片所在根目录路径**（`input_path`）；\n"
        "2）调用名为 `Infer And Save Layout` 的工具，将上一步返回的 `input_path` 作为 `target_path` 传入。该工具会对这些图片执行版面分析、自动保存 classification 结果，并返回完整的 `classification` 数组；\n"
        "3）检查上一步返回的 classification 结果，如果发现有任何图片的 modality 为 \"ocr+figure\"，**请立即调用名为 `prepare_and_launch_validation_gui` 的工具**（不需要任何参数）启动验证窗口；\n"
        "   - 如果工具返回状态为 'skipped'（消息包含'已完成验证'），说明验证已完成，直接继续第 4 步；\n"
        "4）当工具返回成功后，**立即**调用名为 `Organize Dataset By Modality` 的工具，根据验证后的结果将图片分类整理到 `data` 目录；\n"
        "   - **重要**：必须传入参数 `with_segmentation=True`，以执行分割操作裁剪 figure/table/text 区域；\n"
        "   - 可以通过 `min_ocr_area_ratio` 和 `min_figure_area_ratio` 分别控制文本/图表切片的最小过滤比例；\n"
        "   - 如果需要指定验证结果路径，传入 `validated_json_path` 参数；\n"
        "5）最后输出 `data` 目录的整理统计结果（包含分割统计）作为最终报告。\n"
        "\n6）(反思已交由上层主管处理，你只需专注版面分析)"
    )

    full_sys_prompt = f"{base_sys_prompt}\n\n{context_str}"

    agent = ReActAgent(
        name="Modal identification",
        sys_prompt=full_sys_prompt,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        max_iters=20,
    )

    user = UserAgent(name="User")

    msg = Msg(name="system", content="请输入图片路径。", role="system")
    print(f"\n{agent.name}: {msg.content}")

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

            # Run agent in a separate task to support interruption
            import time
            start_time = time.time()

            agent_task = asyncio.create_task(agent(msg))
            try:
                msg = await agent_task
                success = True
            except KeyboardInterrupt:
                await agent.interrupt()
                msg = await agent_task
                success = False

            end_time = time.time()

            print(f"\\n[Agent-1] 本次执行完成，成功状态：{success}。详细反思将由上层协调者独立调用 Memory Agent 处理。")

        except EOFError:
            break


if __name__ == "__main__":
    asyncio.run(main())
