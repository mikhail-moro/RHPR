# Offline runtime for the long-history architecture profile

The SASRec Recent128, HSTU Recent128, Mamba4Rec and FuXi profile uses one pinned
runtime for all four architectures. Mamba executes the official Mamba1 CUDA
selective-scan and causal-convolution extensions. It does not use a Python scan
as a GPU benchmark fallback.

The previous Kaggle runtime had Python 3.13 and PyTorch 2.11.0+cu128. The official
Mamba releases inspected on 2026-10-12 provide Python 3.13 wheels through PyTorch
2.10, but no wheel for PyTorch 2.11. A wheel built against PyTorch 2.10 must not be
assumed ABI compatible with 2.11.

## Pinned binary stack

| Component | Version | Source |
| --- | --- | --- |
| Python | Existing Kaggle CPython 3.13 | No interpreter replacement |
| PyTorch | 2.10.0+cu128, cp313, x86_64 | Official PyTorch CUDA 12.8 index |
| Mamba | 2.3.1, CUDA 12, torch2.10, CXX11 ABI true, cp313 | Official state-spaces release |
| causal-conv1d | 1.7.0, CUDA 12, torch2.10, CXX11 ABI true, cp313 | Official Dao-AILab release |
| Triton | 3.6.0 | PyTorch's exact requirement |
| Transformers | 4.51.3 | Mamba's generation utility imports |
| NumPy | 2.2.6 | Explicit, stable version for Python 3.13 |

The bundle includes every active PyTorch CUDA dependency and every active
transitive Python dependency: 47 wheels, 4,897,856,328 bytes compressed and
8,295,394,579 bytes of uncompressed wheel contents. Dependency markers were
evaluated for CPython 3.13 on Linux x86_64; all 66 active dependency edges are
satisfied. The build does not assume that Kaggle's existing NVIDIA libraries
have the requested versions.

Mamba 2.3.1 avoids the additional Mamba3 package imports and dependencies present
in 2.3.2.post1. The Mamba1 module and selective-scan interface in 2.3.1 are
byte-identical to the previously inspected 2.2.6.post3 files. This choice does not
change the Mamba1 architecture.

The official native wheel filenames include a local CUDA/ABI version suffix,
but their embedded package metadata has the plain package version. pip rejects
that filename/metadata mismatch. The two files are renamed to their embedded
versions, without changing their bytes:

| Mounted filename | SHA256 |
| --- | --- |
| `mamba_ssm-2.3.1-cp313-cp313-linux_x86_64.whl` | `f30e837dcc02f118e29e6502ed09cfcfb6ea3b76c78e6b849f39dd6138ecde04` |
| `causal_conv1d-1.7.0-cp313-cp313-linux_x86_64.whl` | `ca4b91cb5df8a561cd046902cb9cfa749f4cef04e9c9d9d870762f0dc5b23f0c` |
| `torch-2.10.0+cu128-cp313-cp313-manylinux_2_28_x86_64.whl` | `7b4bd23ed63de97456fcc81c26fea9f02ee02ce1112111c4dac0d8cfe574b23e` |

## Offline installation

Attach the private runtime Dataset as an explicit experiment data source. Find
its `runtime-manifest.json` under `/kaggle/input`, verify the manifest and wheel
hashes, and install before importing torch, numpy, transformers or the model
modules. Use a fresh target directory and the existing Python 3.13 interpreter:

```python
import os
import subprocess
import sys
from pathlib import Path

wheel_root = Path("/kaggle/input/rhpr-cp313-torch210-mamba-runtime")
runtime_site = Path("/kaggle/working/rhpr-runtime-site")
install_tmp = Path("/kaggle/working/rhpr-runtime-install-tmp")
install_tmp.mkdir(parents=True, exist_ok=True)
environment = os.environ.copy()
environment["TMPDIR"] = str(install_tmp)
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "--no-index", "--find-links", str(wheel_root),
    "--require-hashes", "--only-binary=:all:",
    "--ignore-installed", "--no-cache-dir", "--no-compile",
    "--target", str(runtime_site),
    "-r", str(wheel_root / "requirements-lock.txt"),
], env=environment, check=True)
sys.path.insert(0, str(runtime_site))
```

The concrete mount path can have additional Kaggle directory levels; resolve
the declared Dataset through its manifest instead of relying on a universal
path layout. Do not add an online pip index or permit source builds.

Setting `TMPDIR` to a disk-backed working directory matters: pip stages an entire
target installation before moving it into place. A small `/tmp` tmpfs cannot
hold this 8.3 GB runtime and can also consume host RAM. Runtime installation,
checksum verification and import checks are setup costs and are recorded
separately from timed model steps.

`requirements-lock.txt` has SHA256
`7403aec00eb0ce74b03e06b062a4e382d54e59e2b9fff94f9f84f949fc6521cb`.
`runtime-manifest.json` has SHA256
`cb1cd1421931548762bd02636771e9927ae7e6993a4dfe7df51810a874baa643`.
Every wheel hash matches its official release asset digest or the PyPI/PyTorch
index checksum. Licenses remain in the wheels and are also extracted alongside
the manifest; Tokenizers' omitted wheel license is supplied from its official
v0.21.4 tag. A runtime Dataset must preserve these component licenses; it cannot
relicense the binary collection as CC0 or Apache-2.0.

## Validation boundary

The complete offline, hash-locked installation passed on a CPU host using
CPython 3.13.5. PyTorch reported 2.10.0+cu128 and CXX11 ABI true; both native
extensions imported with their forward/backward functions, and the standard
Mamba1 class constructed successfully. Import and construction took about eight
seconds. This checks the Python and PyTorch ABI and shared-library linkage;
it does not validate GPU execution. The authorized
Kaggle profile must run a native CUDA forward and backward smoke check on T4
before reporting Mamba timings. All four architectures must report the actual
PyTorch, CUDA, GPU and runtime-manifest identity used in the timed run.

## Primary sources

- [Mamba v2.3.1 release](https://github.com/state-spaces/mamba/releases/tag/v2.3.1)
- [Mamba v2.3.2.post1 release](https://github.com/state-spaces/mamba/releases/tag/v2.3.2.post1)
- [Mamba v2.3.1 wheel selection and dependencies](https://github.com/state-spaces/mamba/blob/v2.3.1/setup.py)
- [causal-conv1d v1.7.0 release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0)
- [Official PyTorch CUDA 12.8 wheel index](https://download.pytorch.org/whl/cu128/torch/)
- [Kaggle Python image Dockerfile](https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl)
- [Tokenizers v0.21.4 license](https://github.com/huggingface/tokenizers/blob/v0.21.4/LICENSE)
