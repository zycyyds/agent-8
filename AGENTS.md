# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## 常用命令

### 运行主编排器（推荐）

```bash
source cs
```

或手动：

```bash
export OLLAMA_HOST=http://<your-ollama-host>:11434
cd main_orchestrator
conda activate py310
python main_orchestrator.py
```

### 运行记忆架构演示

```bash
cd memory_agent
python main.py
```

### 单独运行 Step6+7（独立流水线）

```bash
cd agent_6-7
python run_step6_7_ml_pipeline.py
```

## 架构概览（当前真实状态）

- **main_orchestrator**：主入口，负责串联 Step1→Step2_3→Step4→Step5→Step6→Step7。
- **memory_agent**：每步骤后接收 trace 并更新 `memory_bank.json`。
- **agent_1**：Step1 模态识别与版面分析。
- **agent_2-3**：Step2_3 执行引擎（已接入，含 liver notes 专项分支）。
- **agent_4**：Step4 数据质量修复（已接入）。
- **agent_5**：Step5 任务导向列筛选（已接入）。
- **agent_6-7**：Step6+7 完整实现（已可由主编排调用，也可独立运行）。

## 主链路输出与 next_input 契约

- Step2_3 结果：`program/output/step2_3_results`
  - 发布给 Step4：`program/output/step2_3_results/next_input/input.csv`
- Step4 结果：`program/output/step4_results`
  - 发布给 Step5：`program/output/step4_results/next_input/input.csv`
- Step5 结果：`program/output/step5_results`
  - 常规输出：`*_filtered_时间戳.csv`、`*_selection_report_时间戳.json`
  - 发布给 Step6/7：
    - `program/output/step5_results/next_input/filtered.csv`
    - `program/output/step5_results/next_input/selection_report.json`

Step4 / Step5 / Step6 在读取上游产物时，均采用“**next_input 优先，旧扫描逻辑回退**”。

## Step6_7 目录说明

`agent_6-7` 主要文件：

- `run_step6_7_ml_pipeline.py`：Step6+7 一体化主脚本
- `build_icd10_vector_store.py`：ICD-10 BERT 向量库离线构建
- `download_bert_modelscope.py`：从 ModelScope 下载句向量模型
- `run_step6.py` / `run_step7_phenotyping.py`：历史/旁路脚本，不是当前主入口
- `ml_data_processor_agent.py`：Step7 后处理（将 ml_dataset 转为更适合建模的 processed 数据）

## 注意事项

- Step2_3 在检测到 liver jsonl 时，优先走 liver 专项批处理分支。
- 文档与代码冲突时，以代码为准（优先查看 `main_orchestrator/step_wrappers.py`）。
