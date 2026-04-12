# Multi-Agent Medical Data Pipeline

这是一个面向医疗数据处理的多智能体项目，当前代码以 `main_orchestrator` 为主编排入口，串联 Step1~Step5，并在每个步骤后接入 `memory_agent` 反思闭环。

## 当前状态（以代码为准）

- 已接入主链路：**Step1 → Step2_3 → Step4 → Step5**
- `memory_agent` 已接入主链路：每步骤执行后都会回传 trace 做反思
- `agent_6-7` 已有完整实现，但**目前未接入主编排器**（主编排中的 Step6/7 仍是占位）

## 目录概览

```text
agent_1/                          # Step1 模态识别与版面分析
agent_4/                          # Step4 数据质量修复
agent_5/                          # Step5 任务导向列筛选
main_orchestrator/                # 主编排器与步骤包装层
memory_agent/                     # 反思与记忆库（playbook / memory_bank）
agent_2-3/                        # Step2_3 真实执行引擎（已接入）
agent_6-7/                        # Step6+7 独立流水线（未接入主编排）
program/output/                    # 主链路中间产物
output/                            # Step1 产物等
rawdata/                           # 原始输入数据
cs                                 # 常用启动脚本
```

## 快速开始

### 1) 运行主编排器（推荐）

```bash
source cs
```

`cs` 会设置模型环境变量、激活 conda 环境并启动 `main_orchestrator/main_orchestrator.py`。

也可手动运行：

```bash
export OLLAMA_HOST=http://<your-ollama-host>:11434
cd main_orchestrator
conda activate py310
python main_orchestrator.py
```

### 2) 运行记忆模块演示

```bash
cd memory_agent
python main.py
```

### 3) 单独运行 Step6+7（独立流水线）

```bash
cd agent_6-7
python run_step6_7_ml_pipeline.py
```

> 说明：该脚本会顺序执行 Step6 和 Step7，但当前不会被 `main_orchestrator` 自动调用。

## 主编排链路输入输出

- Step1 输入：用户提供的文件/目录路径
- Step1 输出：`program/output/data`（供 Step2_3 读取）
- Step2_3 输出：`program/output/step2_3_results`
- Step4 输出：`program/output/step4_results`
- Step5 输出：`program/output/`
  - `*_filtered_时间戳.csv`
  - `*_selection_report_时间戳.json`

## Step6_7 独立流水线输入输出

目录：`agent_6-7`

- 输入（Original 模式）：
  - `data/副本all_patients.csv`
  - `data/副本all_patients_filtered_*.csv`（自动取最新）
  - `data/副本all_patients_selection_report_*.json`（自动取最新）
- 输入（MIMIC 模式）：
  - `data/mimic/liver_notes_extracted_filtered*.csv`
  - `data/mimic/liver_notes_extracted_selection_report*.json`
- ICD 标准库：优先读取环境变量 `ICD10_XLSX`；未设置时按脚本内候选路径自动查找
- 输出：
  - Step6：`output_step6/` 或 `output_step6_mimic/`
  - Step7：`output_step7/` 或 `output_step7_mimic/`

## 备注

- 目前仓库未提供统一的测试 / lint 命令入口。
- 文档若与代码不一致，请优先以 `main_orchestrator/step_wrappers.py` 与各步骤主脚本为准。
