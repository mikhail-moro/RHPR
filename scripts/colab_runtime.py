"""Entry point copied to a fresh Colab VM by scripts/colab_job.py."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    workspace = Path(__file__).resolve().parent
    config_path = workspace / "experiment.json"
    config = json.loads(config_path.read_text())
    output = Path("/kaggle/working")
    runtime = config.get("runtime")

    if runtime and len(sys.argv) == 1:
        # The current Colab image need not ship the Python minor used to build
        # the experiment's hash-locked Kaggle wheels.
        uv = shutil.which("uv")
        if not uv:
            subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.10.12"], check=True)
            uv = shutil.which("uv")
        venv = workspace / "venv"
        subprocess.run([uv, "venv", "--python", runtime["target_python_minor"], "--seed", str(venv)], check=True)
        python = str(venv / "bin/python")
        if runtime["kind"] == "gr":
            lock = Path(runtime["wheel_path"]) / "runtime-manifest.json"
            version = json.loads(lock.read_text())["expected_torch_version"].split("+")[0]
            subprocess.run([uv, "pip", "install", "--python", python, "--index-url",
                            "https://download.pytorch.org/whl/cu128", f"torch=={version}+cu128"], check=True)
        os.execv(python, [python, "-u", str(__file__), "--runtime-ready"])

    sys.path.insert(0, str(workspace / "src"))
    (output / "colab-runtime.json").write_text((workspace / "deployment.json").read_text())
    os.environ["GITHUB_SHA"] = json.loads((workspace / "deployment.json").read_text())["git_sha"]
    if runtime:
        from rhpr.offline_runtime import install_offline_runtime
        install_offline_runtime(runtime, output)

    if config.get("execution", {}).get("colab_gpu"):
        import torch
        from rhpr.gr.ml1m_devices import devices_for

        devices_for(config)
        hardware = json.loads((output / "colab-runtime.json").read_text())
        hardware.update(torch=torch.__version__, cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(0), gpu_driver=subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip())
        (output / "colab-runtime.json").write_text(json.dumps(hardware, indent=2) + "\n")

    from rhpr.runner import execute
    execute(config_path, output)


if __name__ == "__main__":
    main()
