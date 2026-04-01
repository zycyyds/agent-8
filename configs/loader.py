import os
import yaml
from pathlib import Path

def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent

def load_model_config():
    """读取并解析模型配置文件"""
    config_path = get_project_root() / "configs" / "model_config.yaml"
    
    if not config_path.exists():
        # 如果配置文件不存在，返回一个空的结构或者报错
        print(f"警告: 配置文件 {config_path} 不存在，将使用代码中的默认值。")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"错误: 解析配置文件失败: {e}")
            return {}

# 导出配置
model_config = load_model_config()

def get_agent_config(agent_key: str):
    """根据智能体 key 获取其特定配置"""
    agents_cfg = model_config.get("agents", {})
    return agents_cfg.get(agent_key, {})
