# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 常用命令

### 运行主布局分析智能体（带记忆闭环）
```bash
cd agent-1
python main_layout_example.py
```

### 运行记忆架构演示
```bash
cd memory_agent
python main.py
```

### 运行主编排器（Ollama + conda 环境）
```bash
export OLLAMA_HOST=http://100.106.134.57:11434
cd main_orchestrator
conda activate py310
python main_orchestrator.py
```

> 目前未在 README 中记录测试或 lint 命令。

## 架构概览

- **agent_1**：Generator（执行器）。负责调用视觉模型进行文档/图像版面分析与模态识别，入口为 `agent_1/main_layout_example.py`。输出结果写入项目根目录的 `output/` 与 `data/` 相关路径。
- **memory_agent**：Reflector + Curator。通过 `memory_agent/memory_tool.py` 对外提供：运行前检索 Playbook 注入上下文、运行后消费 trace 并产出增量反思，再由 Curator 归并更新 `memory_bank.json`。
- **agent_1/doclayout_yolo**：视觉检测模型依赖与结构目录，被 `agent_1` 侧用于版面检测。
- **main_orchestrator**：顶层编排入口（见 `cs` 脚本），负责组织整体流程与模型服务连接。

### 关键数据流（ACE）
1. `agent_1` 在每次处理前，从 `memory_agent` 侧拉取历史策略（Playbook context）。
2. 视觉分析完成后，`agent_1` 将执行痕迹回传给 `memory_agent`。
3. `memory_agent` 生成结构化反思增量并合并进 `memory_bank.json`，形成可持续改进的策略库。
