"""Verified GenPAS components; the original training launcher is not required."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import shutil
import sys
import tempfile

GENPAS_REVISION = "7803b541f646237084ac01692accc8e97053491b"
SOURCE_SHA256 = "3b62260324a76f21062b3fe363023f1277d9f952c19093ea4f016704de827f11"
CONFIG_SHA256 = "35b057a82fd105e801d7adf7c54992fe0da82724d169224cf6ff7245b3fc8cd7"


def tree_digest(root, folder, suffix):
    root = Path(root)
    entries = [p.relative_to(root).as_posix() + "\0" + hashlib.sha256(p.read_bytes()).hexdigest() + "\n"
               for p in sorted((root / folder).rglob("*" + suffix))]
    return hashlib.sha256("".join(entries).encode()).hexdigest()


def prepare_source(source, destination=None):
    """Verify source/configs and defer only the annotation-only TF import.

    Preserve Snap's license. Never alter the model, sampler, loss or beam search.
    A fresh process is required because GRID and GenPAS both import ``src``.
    """
    source = Path(source).resolve()
    if (tree_digest(source, "src", ".py") != SOURCE_SHA256
            or tree_digest(source, "configs", ".yaml") != CONFIG_SHA256):
        raise ValueError(f"GenPAS source/configuration does not match pinned revision {GENPAS_REVISION}")
    target = Path(destination) if destination else Path(tempfile.mkdtemp(prefix="rhpr-genpas-"))
    if destination:
        if target.exists():
            raise ValueError("GenPAS destination must not already exist")
        target.mkdir(parents=True)
    shutil.copytree(source / "src", target / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(source / "configs", target / "configs")
    shutil.copy2(source / "License+Attribution.txt", target)
    interface = target / "src/data/loading/components/interfaces.py"
    original = interface.read_text()
    interface.write_text("from __future__ import annotations\n\n" + original.replace(
        "from typing import Dict, List, Optional, Union, Any",
        "from typing import TYPE_CHECKING, Dict, List, Optional, Union, Any",
    ).replace("from src.data.loading.components.iterators import RawDataIterator",
              "if TYPE_CHECKING:\n    from src.data.loading.components.iterators import RawDataIterator"))
    return target


def activate_source(source):
    if any(name == "src" or name.startswith("src.") for name in sys.modules):
        raise RuntimeError("GenPAS needs a fresh process; another top-level src package is already imported")
    prepared = prepare_source(source)
    sys.path.insert(0, str(prepared))
    importlib.import_module("src.utils.custom_hydra_resolvers")
    return prepared


def compose_beauty(source, data_dir, output_dir):
    """Use the author's Beauty model/optimizer/evaluator, without its launcher."""
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(Path(source) / "configs"), version_base="1.3"):
        return compose(config_name="train", overrides=[
            "experiment=tiger_train_10_00_10_beauty", "trainer=cpu", "logger=csv",
            f"paths.data_dir={Path(data_dir).resolve()}", f"paths.output_dir={Path(output_dir).resolve()}",
            f"paths.work_dir={Path(output_dir).resolve()}", f"paths.profile_dir={Path(output_dir).resolve()}/profile",
        ])
