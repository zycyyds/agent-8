# agents/cleaner_designer_agent.py
from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
import os

def create_cleaner_designer_agent() -> ReActAgent:
    """
    创建一个通用 CSV 数据清洗设计智能体。
    - sys_prompt 为空，行为完全由 user prompt 控制
    - memory=None，避免重试或长链路时上下文累积导致性能下降
    - 可安全复用两套提示词：分析 + cleaner 生成
    """
    agent = ReActAgent(
        name="MedicalDataCleanerExpert",
        sys_prompt="",  # 清空系统提示
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=False,
        ),
        formatter=DashScopeChatFormatter(),
        memory=None,
    )

    return agent
