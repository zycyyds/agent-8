#!/bin/bash
# Medical Data Cleaner 运行脚本

# 设置工作目录
cd "$(dirname "$0")"

# ========== 配置 API Key（请修改为你的实际值）==========
# 方法1: 使用OpenAI API
export OPENAI_API_KEY="your-openai-api-key-here"
export OPENAI_API_BASE="https://api.openai.com/v1"

# 方法2: 使用DashScope（阿里云）
# export DASHSCOPE_API_KEY="your-dashscope-key"
# export OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"

# 方法3: 使用其他兼容OpenAI的API
# export OPENAI_API_KEY="your-api-key"
# export OPENAI_API_BASE="http://your-api-base/v1"

# UMLS API 配置（可选，用于术语标准化）
# 申请地址: https://uts.nlm.nih.gov/uts/signup-login
export UMLS_API_KEY="your-umls-api-key-here"

# ========== 网络代理设置（可选）==========
# 如果需要代理才能访问 UMLS，取消下面注释并填入代理地址
# export http_proxy="http://your-proxy:port"
# export https_proxy="http://your-proxy:port"

# 强制使用代理连接 UMLS（设为true时须确保代理配置可用）
export UMLS_USE_PROXY="false"

# 控制是否使用 UMLS（如果 UMLS 连接超时，设为 "false" 使用 LLM fallback）
export USE_UMLS="true"

# 控制是否优先使用 LOINC（用于量纲统一）
export LOINC_FIRST="false"  # 默认 false，优先使用本地规则

# 详细输出
export UMLS_VERBOSE="true"

# ========== 文件夹过滤设置 ==========
# 遍历目录时跳过的文件夹名称，多个名称用逗号分隔
export SKIP_FOLDERS="figure,分割"

# ========== 使用示例 ==========

# 1. 处理单个图片
# python main.py --input /path/to/your/image.jpg -v

# 2. 处理单个CSV文件（自动检测处理模式）
# python main.py --input /path/to/your/data.csv -v

# 3. 批量处理目录下所有图片
# python main.py --input /path/to/your/data/directory --batch -v

# 4. 交互模式
# python main.py --interactive

# 5. 仅分析不处理
# python main.py --input /path/to/your/file.csv --analyze

# 6. 运行演示
# python main.py --demo

# 7. 处理 liver_patients_note.jsonl（就诊文本信息抽取）
# python main.py --liver-notes /path/to/liver_patients_note.jsonl -v

# ========== 默认运行：交互模式 ==========
echo "启动交互模式..."
echo "提示：直接输入文件路径即可处理，输入 'quit' 退出"
echo ""
python main.py --interactive
