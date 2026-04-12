# Medical Data Cleaner - 医学数据清洗Agent框架

基于AgentScope的医学数据清洗框架，支持多种数据格式的自动处理。

## 项目结构

```
medical_data_cleaner/
├── main.py                     # 主入口
├── run.sh                      # 运行脚本
├── schema.py                   # Pydantic数据模型定义
├── README.md                   # 本文件
│
├── config/                     # 配置
│   ├── __init__.py
│   └── settings.py             # 设置和常量定义
│
├── tools/                      # 工具模块
│   ├── __init__.py
│   ├── data_type_detector.py   # 数据类型检测
│   ├── csv_reader.py           # CSV读取工具
│   ├── tools_ocr.py            # OCR工具
│   ├── tools_preprocess.py     # 预处理工具
│   ├── tools_standardization.py      # 标准化工具（本地知识库）
│   ├── tools_standardization_llm.py  # 标准化工具（LLM增强）
│   └── tools_umls.py           # UMLS API集成
│
├── processors/                 # 数据处理器
│   ├── __init__.py
│   └── csv_processor.py        # CSV处理器
│
├── agents/                     # Agent定义
│   ├── __init__.py
│   ├── data_type_detector_agent.py       # 数据类型检测Agent
│   ├── unified_processing_agent.py       # 统一处理Agent
│   ├── standardization_agent.py          # 标准化Agent
│   └── standardization_agent_llm.py      # LLM增强标准化Agent
│
├── utils/                      # 工具函数
│   └── __init__.py
│
├── examples/                   # 示例
│   └── demo_csv.py
│
└── results/                    # 处理结果保存目录
```

## 支持的数据类型

| 数据类型 | 处理流程 |
|---------|---------|
| **图片** (jpg, png等) | OCR → 预处理 → 信息抽取 → 标准化 → 量纲统一 |
| **文本** (txt等) | 预处理 → 信息抽取 → 标准化 → 量纲统一 |
| **CSV** | 标准化 → 量纲统一 (跳过预处理和信息抽取) |

## 快速开始

### 1. 环境配置

设置环境变量（或在 `run.sh` 中配置）：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="your-api-base"   # 可选，默认OpenAI官方
export UMLS_API_KEY="your-umls-api-key"  # 可选，启用UMLS标准化
```

### 2. 运行

```bash
# 使用 run.sh
./run.sh

# 或直接运行
cd medical_data_cleaner
python main.py --input /path/to/your/file
```

### 3. 命令行参数

```bash
python main.py --help

# 处理单个文件
python main.py --input /path/to/file.csv

# 分析文件结构（不处理）
python main.py --input /path/to/file.csv --analyze

# 批量处理目录
python main.py --input /path/to/directory --batch

# 交互模式
python main.py --interactive

# CSV处理演示
python main.py --demo

# 详细输出
python main.py --input /path/to/file --verbose
```

## 标准化功能

系统使用三级标准化策略：

1. **本地知识库** - 预定义的医学术语映射
2. **UMLS API** - 使用UMLS统一医学语言系统（需配置API密钥）
3. **LLM** - 使用大语言模型进行智能标准化

### 标准化编码系统

- **ICD-10** - 疾病诊断编码
- **ATC** - 药物分类编码
- **LOINC** - 实验室检查编码
- **SNOMED-CT** - 医学术语编码

### 量纲统一

自动将常见单位转换为标准单位：
- 血糖: mg/dL → mmol/L
- 白细胞: /mm³ → ×10⁹/L
- 肌酐: mg/dL → μmol/L
- 等等...

## 依赖

```
agentscope
pydantic
pandas (CSV处理)
opencv-python (图片处理)
requests (UMLS API)
```

## 输出

处理结果自动保存到 `results/` 目录，文件名格式：
```
{原文件名}_processed_{时间戳}.json
```
