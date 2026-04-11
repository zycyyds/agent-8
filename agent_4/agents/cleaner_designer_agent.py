# agents/cleaner_designer_agent.py
import os

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter, OpenAIChatFormatter
from agentscope.model import DashScopeChatModel, OpenAIChatModel


OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_TIMEOUT = 120
OPENAI_MODEL_NAME = "gpt-4.1-mini"
DASHSCOPE_MODEL_NAME = "qwen-max"


def _get_model_and_formatter():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        model = OpenAIChatModel(
            model_name=os.environ.get("MODEL_NAME", OPENAI_MODEL_NAME),
            api_key=openai_api_key,
            client_kwargs={
                "base_url": os.environ.get("OPENAI_API_BASE", OPENAI_API_BASE),
                "timeout": int(os.environ.get("OPENAI_TIMEOUT", str(OPENAI_TIMEOUT))),
            },
            generate_kwargs={"temperature": 0.0},
        )
        return model, OpenAIChatFormatter()

    dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
    if dashscope_api_key:
        model = DashScopeChatModel(
            model_name=os.environ.get("DASHSCOPE_MODEL_NAME", DASHSCOPE_MODEL_NAME),
            api_key=dashscope_api_key,
            stream=False,
        )
        return model, DashScopeChatFormatter()

    raise RuntimeError("未配置可用模型：请设置 OPENAI_API_KEY，或设置 DASHSCOPE_API_KEY。")


def create_cleaner_designer_agent() -> ReActAgent:
    """
    创建一个通用 CSV 数据清洗设计智能体。
    - sys_prompt 为空，行为完全由 user prompt 控制
    - memory=None，避免重试或长链路时上下文累积导致性能下降
    - 可安全复用两套提示词：分析 + cleaner 生成
    """
    model, formatter = _get_model_and_formatter()
    agent = ReActAgent(
        name="MedicalDataCleanerExpert",
        sys_prompt="",
        model=model,
        formatter=formatter,
        memory=None,
    )

    return agent
