export OPENAI_API_KEY="key-2"
export OPENAI_API_BASE="http://127.0.0.1:8317"  
export MODEL_NAME="gpt-5"       

export OLLAMA_HOST=http://100.106.134.57:11434
cd main_orchestrator
conda activate py310
python  main_orchestrator.py                  