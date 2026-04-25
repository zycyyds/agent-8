export OPENAI_API_KEY="key-1"
export OPENAI_API_BASE="http://127.0.0.1:8317/v1"
export MODEL_NAME="gpt-5.4"
export OLLAMA_HOST="http://100.106.134.57:11434"

export ICD10_USE_BERT=1
export ICD10_BERT_MODEL="/Users/mkbk/PycharmProjects/step1-dataset-codegen/agent_6-7/models/bge-small-zh-v1.5"
export ICD10_BERT_MIN_SIM=0.70
export ICD10_XLSX="/Users/mkbk/PycharmProjects/step1-dataset-codegen/agent_6-7/ICD-10.xlsx"
export DATA_SOURCE=original

conda activate py310
cd "/Users/mkbk/PycharmProjects/step1-dataset-codegen"
python main_orchestrator/main_orchestrator.py --disable-memory-agent
