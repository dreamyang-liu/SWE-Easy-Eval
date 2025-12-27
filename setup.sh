
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.9+
uv venv vllm-env --python 3.12

# Activate virtual environment
source vllm-env/bin/activate

# Install vLLM
uv pip install vllm

# Install mini-swe-agent as tool
uv tool install mini-swe-agent --with datasets

