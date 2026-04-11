# agents/cleaner_designer_agent.py
import os

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel


OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_TIMEOUT = 120
OPENAI_MODEL_NAME = "gpt-4.1-mini"


def _get_model_and_formatter():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("未配置可用模型：请设置 OPENAI_API_KEY。")

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


def _strip_thinking_from_msg(msg: Msg) -> Msg:
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return msg
    content_blocks = [
        block for block in msg.get_content_blocks() if str(block.get("type") or "") != "thinking"
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


def _register_no_thinking_print_hook(agent: ReActAgent) -> ReActAgent:
    def _hook(_agent: ReActAgent, kwargs: dict):
        msg = kwargs.get("msg")
        if isinstance(msg, Msg):
            kwargs["msg"] = _strip_thinking_from_msg(msg)
        return kwargs

    agent.register_instance_hook("pre_print", "strip_thinking_for_console", _hook)
    return agent


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

    return _register_no_thinking_print_hook(agent)
