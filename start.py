#!/usr/bin/env python3
"""
SWE-bench Model Evaluation Automation Script
Automates the complete evaluation pipeline from setup to results.
"""

import os
import sys
import json
import subprocess
import argparse
import signal
import time
import psutil
import requests
from pathlib import Path

def run_cmd(cmd, check=True, shell=True):
    """Execute shell command with error handling."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def check_docker():
    """Verify Docker is installed and running."""
    try:
        run_cmd("docker --version")
        print("✓ Docker is available")
    except:
        print("✗ Docker not found. Install from https://docs.docker.com/get-docker/")
        sys.exit(1)

def setup_environment():
    """Run setup script and install dependencies."""
    if Path("setup.sh").exists():
        run_cmd("bash setup.sh")
    else:
        print("Warning: setup.sh not found, skipping...")

def create_config_files(model_name, api_base, benchmark_dir):
    """Create configuration files for the model."""
    benchmark_path = Path(benchmark_dir)
    benchmark_path.mkdir(exist_ok=True)

    # Create swebench.yaml
    config = {
        'model': {
            'model_name': f"hosted_vllm/{model_name}",
            'cost_tracking': "ignore_errors",
            'litellm_model_registry': str(benchmark_path / "registry.json"),
            'model_kwargs': {
                'api_base': api_base
            }
        }
    }

    import yaml
    with open(benchmark_path / "swebench.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Create registry.json
    registry = {
        model_name: {
            "max_tokens": 32768,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "hosted_vllm",
            "mode": "chat"
        }
    }

    with open(benchmark_path / "registry.json", 'w') as f:
        json.dump(registry, f, indent=2)

    print(f"✓ Config files created in {benchmark_dir}")

class VLLMServerManager:
    def __init__(self, model_name, api_base="http://0.0.0.0:8000", tensor_parallel_size=4, port=8000):
        self.model_name = model_name
        self.api_base = api_base
        self.tensor_parallel_size = tensor_parallel_size
        self.port = port
        self.process = None

    def is_port_in_use(self):
        """Check if port is already in use."""
        for conn in psutil.net_connections():
            if conn.laddr.port == self.port:
                return True
        return False

    def kill_existing_servers(self):
        """Kill any existing VLLM servers on the port."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and any('vllm' in cmd for cmd in proc.info['cmdline']):
                    if any(str(self.port) in cmd for cmd in proc.info['cmdline']):
                        print(f"Killing existing VLLM process: {proc.info['pid']}")
                        proc.terminate()
                        proc.wait(timeout=10)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass

    def wait_for_server(self, timeout=300):
        """Wait for server to be ready."""
        print("Waiting for VLLM server to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.api_base}/v1/models", timeout=5)
                if response.status_code == 200:
                    print("✓ VLLM server is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(5)

        print(f"✗ VLLM server not ready after {timeout} seconds")
        return False

    def start(self):
        """Start VLLM server."""
        if self.is_port_in_use():
            print(f"Port {self.port} is in use. Killing existing servers...")
            self.kill_existing_servers()
            time.sleep(5)

        cmd = [
            "vllm", "serve", self.model_name,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--port", str(self.port),
            "--host", "0.0.0.0"
        ]
        print(f"Starting VLLM server: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if not self.wait_for_server():
            self.stop()
            raise RuntimeError("Failed to start VLLM server")

        return True

    def stop(self):
        """Stop VLLM server."""
        if self.process:
            print("Stopping VLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            print("✓ VLLM server stopped")

    def is_running(self):
        """Check if server is running."""
        if not self.process:
            return False
        return self.process.poll() is None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

def generate_predictions(model_name, config_path, output_dir, workers=32):
    """Generate model predictions on SWE-bench."""
    cmd = f"""mini-extra swebench \
        --model hosted_vllm/{model_name} \
        --config {config_path} \
        --subset verified \
        --output {output_dir} \
        --split test \
        --workers {workers}"""

    run_cmd(cmd)
    print(f"✓ Predictions saved to {output_dir}/preds.json")
def setup_swebench():
    """Clone and install SWE-bench evaluation tools."""
    if not Path("SWE-bench").exists():
        run_cmd("git clone https://github.com/SWE-bench/SWE-bench.git")

    run_cmd("cd SWE-bench && pip install -e .")
    print("✓ SWE-bench installed")

def run_evaluation(predictions_path, run_id, max_workers=32):
    """Run the final evaluation."""
    cmd = f"""python -m swebench.harness.run_evaluation \
        --dataset_name SumanthRH/SWE-bench_Verified \
        --predictions_path {predictions_path} \
        --max_workers {max_workers} \
        --run_id {run_id}"""

    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        print("Error: Evaluation command failed")
        sys.exit(1)
    print(f"✓ Evaluation completed for run_id: {run_id}")

def signal_handler(signum, frame, server_manager=None):
    """Handle interrupt signals to cleanup server."""
    print("\nReceived interrupt signal. Cleaning up...")
    if server_manager and server_manager.is_running():
        server_manager.stop()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Automate SWE-bench model evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Model name")
    parser.add_argument("--api-base", default="http://0.0.0.0:8000", help="API base URL")
    parser.add_argument("--port", type=int, default=8000, help="VLLM server port")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--benchmark-dir", default="~/benchmark", help="Benchmark config directory")
    parser.add_argument("--output-dir", default="~/qwen2.5", help="Output directory")
    parser.add_argument("--workers", type=int, default=32, help="Number of workers")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--skip-server", action="store_true", help="Skip VLLM server management")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")

    args = parser.parse_args()

    # Expand paths
    benchmark_dir = Path(args.benchmark_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    print("🚀 Starting SWE-bench evaluation pipeline...")

    # Step 1: Check prerequisites
    check_docker()

    server_manager = None

    try:
        if not args.eval_only:
            # Step 2: Setup environment
            if not args.skip_setup:
                setup_environment()

            # Step 3: Create config files
            create_config_files(args.model, args.api_base, benchmark_dir)

            # Step 4: Manage VLLM server
            if not args.skip_server:
                server_manager = VLLMServerManager(
                    args.model,
                    args.api_base,
                    args.tensor_parallel_size,
                    args.port
                )

                # Setup signal handler for cleanup
                signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, server_manager))
                signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, server_manager))

                server_manager.start()

            # Step 5: Generate predictions
            generate_predictions(
                args.model,
                benchmark_dir / "swebench.yaml",
                output_dir,
                args.workers
            )

        # Step 6: Setup SWE-bench
        setup_swebench()

        # Step 7: Run evaluation
        predictions_path = output_dir / "preds.json"
        if predictions_path.exists():
            run_id = args.model.split('/')[-1].lower().replace('-', '_')
            run_evaluation(predictions_path, run_id, args.workers)
        else:
            print(f"✗ Predictions file not found: {predictions_path}")
            sys.exit(1)

        print("🎉 SWE-bench evaluation completed!")

    except Exception as e:
        print(f"✗ Error during execution: {e}")
        sys.exit(1)

    finally:
        # Cleanup server
        if server_manager and server_manager.is_running():
            server_manager.stop()

if __name__ == "__main__":
    main()