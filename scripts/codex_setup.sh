#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 -m pip install --disable-pip-version-check -e .
python3 -m compileall -q src scripts kaggle tests
python3 -m unittest discover -s tests -v
