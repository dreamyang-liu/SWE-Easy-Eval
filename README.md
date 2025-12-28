# SWE-bench Model Evaluation Guide

A step-by-step guide to evaluate your model on SWE-bench datasets with mini-swe-agent.

## Step 1: Prerequisites

Install Docker and verify it works:
```bash
docker --version
```
If not installed, follow: https://docs.docker.com/get-docker/

## Step 2: Environment Setup

Install required tools:
```bash
bash setup.sh
```

This installs uv, vllm environment, and mini-swe-agent.


# Quick Eval

```shell
conda create -n swe_easy_eval python=3.12 -y
conda activate swe_easy_eval
python start.py --model Qwen/Qwen3-8B --tensor-parallel-size 4 --output-dir ./results/qwen3-8b
```

# Step by step eval

## Step 1: Configure Your Model

Run vllm serve to start a serving server
```shell
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --tensor-parallel-size 4
```

### 1.1 Update Model Configuration
Edit the config file:
```bash
vim ~/benchmark/swebench.yaml
```

Set your model details:
```yaml
model:
  model_name: "hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct"
  cost_tracking: "ignore_errors"
  litellm_model_registry: "~/benchmark/registry.json"
  model_kwargs:
    api_base: "http://0.0.0.0:8000/v1"
```

### 1.2 Update Model Registry
Create `~/benchmark/registry.json`:
```json
{
  "Qwen/Qwen2.5-Coder-7B-Instruct": {
    "max_tokens": 32768,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "hosted_vllm",
    "mode": "chat"
  }
}
```

## Step 2: Generate Predictions

Run the model on SWE-bench:
```bash
mini-extra swebench \
    --model hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct \
    --config ~/benchmark/swebench.yaml \
    --subset verified \
    --output ~/qwen2.5 \
    --split test \
    --workers 32
```

This creates `~/qwen2.5/preds.json` with your model's predictions.

## Step 3: Evaluate Results

### 4.1 Install SWE-bench
```bash
git clone https://github.com/SWE-bench/SWE-bench.git
cd SWE-bench
pip install -e .
```

### 4.2 Run Evaluation
```bash
python -m swebench.harness.run_evaluation \
    --dataset_name SumanthRH/SWE-bench_Verified \
    --predictions_path ~/qwen2.5/preds.json \
    --max_workers 32 \
    --run_id qwen2.5
```