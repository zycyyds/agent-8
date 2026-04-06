export OPENAI_API_KEY="key-2"
export OPENAI_API_BASE="http://127.0.0.1:8317"   # 如果你不是直连官方 OpenAI，就配这个
export MODEL_NAME="gpt-5"       # 可选，不配也有默认值

export OLLAMA_HOST=http://100.106.134.57:11434
cd main_orchestrator
conda activate py310
python  main_orchestrator.py                  