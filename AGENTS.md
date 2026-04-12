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

- **main_orchestrator**：主入口，负责串联 Step1→Step2_3→Step4→Step5。
- **memory_agent**：每步骤后接收 trace 并更新 `memory_bank.json`。
- **agent_1**：Step1 模态识别与版面分析。
- **agent_2-3**：Step2_3 执行引擎（已接入）。
- **agent_4**：Step4 数据质量修复（已接入）。
- **agent_5**：Step5 任务导向列筛选（已接入）。
- **agent_6-7**：Step6+7 完整实现，当前未接入主编排（需单独运行）。

## 主链路输出目录约定

- Step2_3 输出：`program/output/step2_3_results`
- Step4 输出：`program/output/step4_results`
- Step5 输出：`program/output/`
  - `*_filtered_时间戳.csv`
  - `*_selection_report_时间戳.json`

## Step6_7 目录说明

`agent_6-7` 主要文件：

- `run_step6_7_ml_pipeline.py`：Step6+7 一体化主脚本
- `build_icd10_vector_store.py`：ICD-10 BERT 向量库离线构建
- `download_bert_modelscope.py`：从 ModelScope 下载句向量模型
- `run_step6.py` / `run_step7_phenotyping.py`：历史/旁路脚本，不是当前主入口
- `ml_data_processor_agent.py`：Step7 后处理（将 ml_dataset 转为更适合建模的 processed 数据）

## 注意事项

- `main_orchestrator` 中 Step6/7 目前是占位实现，不会自动调用 `step6_7_ml_pipeline`。
- 文档与代码冲突时，以代码为准（优先查看 `main_orchestrator/step_wrappers.py`）。
