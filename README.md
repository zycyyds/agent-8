# Multi-Agent Medical Data Pipeline

这是一个面向医疗数据处理的多智能体项目。当前以 `main_orchestrator` 为主入口，串联 Step1~Step7，并在每个步骤后接入 `memory_agent` 做反思闭环。

## 当前状态（以代码为准）

- 已接入主链路：**Step1 → Step2_3 → Step4 → Step5 → Step6 → Step7**
- `memory_agent` 已接入主链路：每步骤执行后都会回传 trace 做反思
- `agent_6-7` 既可被主编排调用，也可独立运行 `run_step6_7_ml_pipeline.py`

## 目录概览

```text
agent_1/                           # Step1 模态识别与版面分析
agent_2-3/                         # Step2_3 执行引擎
agent_4/                           # Step4 数据质量修复
agent_5/                           # Step5 任务导向列筛选
agent_6-7/                         # Step6+7 实现与独立入口
main_orchestrator/                 # 主编排器与步骤包装层
memory_agent/                      # 反思与记忆库（playbook / memory_bank）
program/output/                    # 主链路中间产物
output/                            # Step1 历史输出
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

> 说明：独立脚本会按其自身参数与目录约定运行，不依赖主编排会话。

## 主链路输入输出与衔接约定

### Step1
- 输入：用户提供的文件/目录路径
- 输出：`program/output/data`（供 Step2_3 读取）

### Step2_3
- 结果目录：`program/output/step2_3_results`
- 向 Step4 发布：`program/output/step2_3_results/next_input/input.csv`
- 处理策略：若检测到 liver jsonl，则优先走 liver 专项；否则进入通用交互处理

### Step4
- 结果目录：`program/output/step4_results`
- 优先读取：`step2_3_results/next_input/input.csv`
- 向 Step5 发布：`program/output/step4_results/next_input/input.csv`

### Step5
- 结果目录：`program/output/step5_results`
  - `*_filtered_时间戳.csv`
  - `*_selection_report_时间戳.json`
- 向 Step6/7 发布：
  - `program/output/step5_results/next_input/filtered.csv`
  - `program/output/step5_results/next_input/selection_report.json`

### Step6 / Step7
- 在主编排中优先读取 `step5_results/next_input/`；缺失时回退到历史命名扫描逻辑
- 主编排输出目录：
  - Step6: `program/output/step6-7_results/step6`
  - Step7: `program/output/step6-7_results/step7`

## Step6_7 独立流水线补充说明

目录：`agent_6-7`

- 入口：`run_step6_7_ml_pipeline.py`
- ICD-10 词表：优先读取环境变量 `ICD10_XLSX`，未设置时按脚本内候选路径查找 `ICD-10.xlsx`
- 可选向量检索：`build_icd10_vector_store.py` 离线构建向量库到 `agent_6-7/data/icd10_bert_index/`

## 备注

- 文档与代码冲突时，以代码为准，优先查看 `main_orchestrator/step_wrappers.py`。
- 目前仓库未提供统一测试/lint 入口命令。
