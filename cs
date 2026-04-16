export OPENAI_API_KEY="key-1"
export OPENAI_API_BASE="http://127.0.0.1:8317/v1"     
export MODEL_NAME="gpt-5.4-mini" 
export STEP2_3_AUTO_LIVER_NOTES=false    
conda activate py310
export SKIP_FOLDERS="figure,files"

export OLLAMA_HOST=http://100.106.134.57:11434
cd main_orchestrator
python  main_orchestrator.py                  