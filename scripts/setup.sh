# create conda environment
conda env create -f env.yml
conda activate llm
# install ollama for local LLM usage
curl -fsSL https://ollama.com/install.sh | sh
ollama serve