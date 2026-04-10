---
name: agent_4 integration snapshot
description: Snapshot of agent_4 standalone CSV cleaner pipeline and its current non-integration with main_orchestrator step wrappers
type: project
---
agent_4 当前是一个独立运行的 CSV 列级 cleaner 生成与执行流水线，不在 main_orchestrator 的真实调用链中。
**Why:** 代码显示 `/Users/mkbk/PycharmProjects/agent-8/agent_4/main.py` 自己读取 `agent_4/data/*.csv`，动态生成 `cleaners/<列名>/implementation.py`，串行执行并输出清洗后 CSV 与验证报告；而 `/Users/mkbk/PycharmProjects/agent-8/main_orchestrator/step_wrappers.py` 的 Step4 仍是 mock，没有 import 或调用 agent_4。
**How to apply:** 后续若用户问 agent_4 接入状态，应先按“独立子流水线、未接入 orchestrator”解释；若要接入，重点检查输入目录适配、工作目录依赖、返回结构标准化、ToolResponse 封装与 trace/memory 回传。