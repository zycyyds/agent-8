# Multi-Agent Layout Analysis System 🧠

这是一个基于 **AgentScope** 和 **ACE (Agentic Context Engineering)** 架构实现的多智能体文档版面分析系统。

本系统不仅能够自动化处理图像/文档的布局识别和分类，还具备**自我反思与记忆（Memory Agent）**的能力，能够在一次次执行中提取经验规则（Bullets），并在下一次处理中变得更聪明。

## 🌟 核心特性

- **视觉版面分析**：基于 YOLO 等视觉模型（封装在 `agent-1` 中），对输入的图像提取表格、图表、OCR 文本块。
- **人机协同验证**：智能体会遇到复杂长文本或包含图表的页面，并自动弹窗请求人类介入复核。
- **ACE 三角色反思架构**：
  - **Generator（生成器/执行器）**：即 `agent-1`。获取之前积累的最佳实践（Context Playbook），根据提示词行动。
  - **Reflector（反思器）**：在每次成功或失败的视觉分析后，反思哪些经验起到了作用，哪些产生了误导，并提取新的经验规则。
  - **Curator（整理器）**：提纯反思器得到的新经验（Delta），合并到持久化的长期记忆库 `memory_bank.json` 中，并去除低效规则。

## 📂 目录结构

```text
├── agent-1/                     # Generator 智能体 (布局分析工具集与主循环)
│   ├── main_layout_example.py   # 🏆 主程序入口 (日常运行只运行此文件)
│   ├── layout_analysis_tool.py  # 版面分析的具体视觉推理库
│   └── agentscope_tools.json    # AgentScope 工具定义
├── memory_agent/                # Reflector & Curator 智能体记忆系统
│   ├── core/
│   │   ├── playbook.py          # 维护原子化策略卡片 (Bullets) 及 Playbook 结构
│   │   ├── reflector.py         # 增量反思，评估策略卡片有效性
│   │   └── curator.py           # 合并 Delta, 整理和精简记忆库
│   ├── integration.py           # 对外提供注入 Context 和回传 Record 的接口
│   └── main.py                  # (测试用) 纯架构演示脚本
├── doclayout_yolo/              # 底层依赖的视觉检测模型结构
└── data/                        # 数据集源及运行中产生的文件缓存
```

## 🚀 快速开始

### 运行主智能体（带记忆能力）

在真实场景下，你只需要运行 `agent-1` 即可，**记忆闭环会在后台自动发生**：

```bash
cd agent-1
python main_layout_example.py
```

- 在命令行中输入图像或目录的路径。
- 智能体会自动：
  1. 调用 `memory_agent` 提取当前的最佳策略。
  2. 调用图像分割、推理工具处理图片。
  3. 执行完毕后将反馈回传给记忆系统，反思并记录本次表现，持久化到 `memory_agent/memory_bank.json` 中。

### 运行记忆架构演示 (仅供测试原理)

如果你想了解 ACE (Generator -> Reflector -> Curator) 三角色的运作时序图与处理结果，可以独立运行演示脚本：

```bash
cd memory_agent
python main.py
```

## 🧠 记忆系统数据流 (ACE Framework)

1. **`get_playbook_context()`**：`agent-1` 在每次拿到文件后，第一时间从这里拉取 `#历史积累的策略与经验提示`，喂给系统提示词（`sys_prompt`）。
2. **`record_from_first_agent(...)`**：`agent-1` 处理完全部工作（无论是异常退出还是成功分析），把所有的痕迹传回这里。
3. **`Reflector.analyze()`**：大模型反思这轮痕迹：“上一次的规矩有帮到忙吗？有哪些没用？这次有没有领悟出新规矩？” -> 产生一个 `ReflectionDelta` (增量)。
4. **`Curator.process_new_execution()`**：将这些增量计数加到具体的 `Bullet`（策略卡片）中，一旦新知识经过反复验证有效，会持续为后续任务保驾护航。

---
*Powered by AgentScope & Ollama*
