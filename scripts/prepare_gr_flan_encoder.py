"""Prepare hash-pinned Flan-T5-XL encoder weights without loading the decoder.

The source shard is verified in full. Tensor bytes are then copied directly to
a valid encoder-only safetensors file, preserving the published FP32 values.
This is CPU/network preparation, not an embedding or training experiment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import struct
import threading
import time
import urllib.request


MODEL = "google/flan-t5-xl"
REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"
SHARD = "model-00001-of-00002.safetensors"
SOURCE = {
    SHARD: (9449619912, "99196ddfbe886e8ef860f52de979df64890edfc792c3d94ce0502991f347dd18"),
    "config.json": (1438, "ed9df0075dda91de9e9431f193203a3f869189e4effa3fc00cf9d41864f514ba"),
    "tokenizer_config.json": (2539, "ef30e0628a2d067c5d9d02d98cbf9a2040f4a51883e5d16e1809eea548db0eff"),
    "tokenizer.json": (2424064, "fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a"),
    "special_tokens_map.json": (2201, "5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f"),
    "spiece.model": (791656, "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"),
    "model.safetensors.index.json": (53032, "4af536474e3441bab3996748fcf730cd47bb4bd66842efda0acdcfdcaa7c7429"),
    "README.md": (10747, "c2210a137e81d30d03f2176cf6fba8132793d2f0286c38d944bd1b0122366515"),
    "LICENSE": (11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
}
NOTICE_URLS = {
    "README.md": f"https://huggingface.co/{MODEL}/resolve/{REVISION}/README.md",
    "LICENSE": "https://www.apache.org/licenses/LICENSE-2.0.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, size: int, digest: str) -> None:
    if path.stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"published file digest mismatch: {path.name}")


def download_notice(url: str, destination: Path, size: int, digest: str) -> None:
    """Retain small upstream notices, also supporting non-range HTTP servers."""
    if destination.exists():
        verify(destination, size, digest)
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read(size + 1)
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"published notice digest mismatch: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_bytes(content)
    temporary.replace(destination)


def download_verified(url: str, destination: Path, size: int, digest: str,
                      *, chunk_size: int = 32 << 20, workers: int = 4) -> None:
    """Resume explicit HTTP ranges, then verify the publisher's whole-file hash."""
    if destination.exists():
        verify(destination, size, digest)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if chunk_size <= 0 or workers <= 0:
        raise ValueError("chunk size and workers must be positive")
    partial = destination.with_name(destination.name + ".partial")
    state_path = destination.with_name(destination.name + ".chunks.json")
    identity = {"size_bytes": size, "sha256": digest, "chunk_size": chunk_size}
    if partial.exists() and state_path.exists():
        saved = json.loads(state_path.read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("partial download belongs to a different published file")
        if partial.stat().st_size != size:
            raise ValueError("partial download allocation differs from published size")
        completed = set(saved["completed"])
    else:
        completed = set()
        with partial.open("wb") as stream:
            stream.truncate(size)
    count = math.ceil(size / chunk_size)
    if any(type(i) is not int or not 0 <= i < count for i in completed):
        raise ValueError("invalid partial chunk index")
    lock = threading.Lock()
    started = time.monotonic()

    def fetch(index):
        start = index * chunk_size
        end = min(size, start + chunk_size) - 1
        # Distinct cache keys avoid a proxy reusing a differently ranged body.
        ranged_url = url + ("&" if "?" in url else "?") + f"rhpr_range={start}-{end}"
        for attempt in range(3):
            try:
                request = urllib.request.Request(ranged_url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("server did not return the requested immutable range")
                    body = response.read(end - start + 2)
                if len(body) != end - start + 1:
                    raise ValueError("truncated or oversized range response")
                with partial.open("r+b") as stream:
                    stream.seek(start)
                    stream.write(body)
                    stream.flush()
                with lock:
                    completed.add(index)
                    temporary = state_path.with_name(state_path.name + ".tmp")
                    temporary.write_text(json.dumps({**identity, "completed": sorted(completed)}))
                    temporary.replace(state_path)
                    if len(completed) % 16 == 0 or len(completed) == count:
                        print(json.dumps({"phase": "download", "file": destination.name,
                                          "chunks": len(completed), "total_chunks": count,
                                          "elapsed_seconds": time.monotonic() - started}), flush=True)
                return
            except Exception:
                if attempt == 2:
                    raise
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, index) for index in range(count) if index not in completed]
        for future in as_completed(futures):
            future.result()
    verify(partial, size, digest)
    partial.replace(destination)
    state_path.unlink(missing_ok=True)


def extract_encoder(source: Path, destination: Path) -> dict:
    """Write a safetensors subset with bounded memory and byte-identical values."""
    with source.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= 64 << 20 or length + 8 > source.stat().st_size:
            raise ValueError("invalid safetensors header length")
        header = json.loads(stream.read(length))
    selected = {name: entry for name, entry in header.items()
                if name.startswith("encoder.") or name.startswith("shared.")}
    if not selected or "shared.weight" not in selected:
        raise ValueError("source does not contain complete encoder/shared tensors")
    body_start = 8 + length
    # The published index names tied embeddings that safetensors stores only
    # once. Preserve the source's encoder alias declaration so validation and
    # Transformers can distinguish deduplication from a missing real weight.
    metadata = {"format": "pt"}
    metadata.update({name: target for name, target in header.get("__metadata__", {}).items()
                     if name.startswith("encoder.") and target in selected})
    packed, offset, tensors = {"__metadata__": metadata}, 0, []
    for name in sorted(selected):
        entry = selected[name]
        shape, positions = entry["shape"], entry["data_offsets"]
        if (entry["dtype"] != "F32" or any(type(x) is not int or x < 1 for x in shape)
                or len(positions) != 2 or any(type(x) is not int for x in positions)):
            raise ValueError("encoder tensors must have valid FP32 shapes and offsets")
        start, end = positions
        expected = math.prod(shape) * 4
        if start < 0 or end - start != expected or body_start + end > source.stat().st_size:
            raise ValueError("encoder tensor offsets differ from shape/source size")
        packed[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + expected]}
        tensors.append({"name": name, "shape": shape, "dtype": "F32", "size_bytes": expected})
        offset += expected
    encoded = json.dumps(packed, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    temporary = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, temporary.open("wb") as outgoing:
        outgoing.write(struct.pack("<Q", len(encoded)))
        outgoing.write(encoded)
        for tensor in tensors:
            start, end = selected[tensor["name"]]["data_offsets"]
            incoming.seek(body_start + start)
            remaining = end - start
            while remaining:
                block = incoming.read(min(8 << 20, remaining))
                if not block:
                    raise ValueError("truncated source tensor payload")
                outgoing.write(block)
                remaining -= len(block)
    temporary.replace(destination)
    return {"tensors": tensors, "tensor_count": len(tensors), "tensor_bytes": offset,
            "parameters": offset // 4, "derived_sha256": sha256(destination)}


def validate_encoder(directory: Path, source_index: Path) -> dict:
    """Validate actual Transformers key completeness without allocating copies."""
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, T5Config, T5EncoderModel

    index = json.loads(source_index.read_text())["weight_map"]
    expected = {name for name in index if name.startswith("encoder.") or name.startswith("shared.")}
    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as weights:
        actual = set(weights.keys())
        aliases = weights.metadata() or {}
        missing = expected - actual
        allowed_alias = {"encoder.embed_tokens.weight"}
        if (actual - expected or missing - allowed_alias
                or any(aliases.get(name) != "shared.weight" or "shared.weight" not in actual
                       for name in missing)):
            raise ValueError("derived tensor keys differ from the pinned full model index")
        state = {name: weights.get_tensor(name) for name in weights.keys()}
    configuration = T5Config.from_pretrained(directory, local_files_only=True)
    with torch.device("meta"):
        model = T5EncoderModel(configuration)
    # Strict loading checks every real model key after restoring the one known
    # tied embedding alias; no newly initialized layer tensors are accepted.
    if "encoder.embed_tokens.weight" not in state:
        state["encoder.embed_tokens.weight"] = state["shared.weight"]
    result = model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    sample = tokenizer("Title: example; ", max_length=128, padding="max_length",
                       truncation=True, add_special_tokens=True, return_tensors="pt")
    if sample["input_ids"].shape != (1, 128):
        raise ValueError("pinned tokenizer does not produce the expected shape")
    return {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
            "d_model": configuration.d_model, "encoder_layers": configuration.num_layers,
            "transformers": importlib.metadata.version("transformers"),
            "safetensors": importlib.metadata.version("safetensors"),
            "tokenizer_class": type(tokenizer).__name__, "tokenizer_sample_tokens": int(sample["attention_mask"].sum()),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "dtype": str(next(model.parameters()).dtype), "embedding_alias_restored": True,
            "index_aliases_restored": sorted(missing)}


def write_manifests(output: Path, report: dict, validation: dict) -> dict:
    """Register prepared bytes; also usable after retaining missing notices."""
    manifest = {"model": MODEL, "revision": REVISION, "derived_encoder_only": True,
                "source_files": {name: {"size_bytes": size, "sha256": digest} for name, (size, digest) in SOURCE.items()},
                "notice_sources": NOTICE_URLS,
                "extraction": report, "validation": validation,
                "files": [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                          for path in sorted(output.iterdir()) if path.is_file()
                          and path.name not in {"encoder-provenance.json", "resources-manifest.json"}]}
    (output / "encoder-provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    resources = {"model": MODEL, "revision": REVISION, "derived_encoder_only": True,
                 "files": [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                           for path in sorted(output.iterdir()) if path.is_file() and path.name != "resources-manifest.json"]}
    (output / "resources-manifest.json").write_text(json.dumps(resources, indent=2, sort_keys=True) + "\n")
    return manifest


def prepare(output: Path, cache: Path, workers: int = 4, keep_source: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"
    for name, (size, digest) in SOURCE.items():
        target = cache / name if name in {SHARD, "model.safetensors.index.json"} else output / name
        if name in NOTICE_URLS:
            download_notice(NOTICE_URLS[name], target, size, digest)
        else:
            download_verified(base + name, target, size, digest, workers=workers)
    report = extract_encoder(cache / SHARD, output / "model.safetensors")
    validation = validate_encoder(output, cache / "model.safetensors.index.json")
    manifest = write_manifests(output, report, validation)
    if not keep_source:
        (cache / SHARD).unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep-source", action="store_true")
    arguments = parser.parse_args()
    result = prepare(arguments.output, arguments.cache, arguments.workers, arguments.keep_source)
    print(json.dumps({"phase": "encoder_prepared", "output": str(arguments.output),
                      "tensor_count": result["extraction"]["tensor_count"],
                      "parameters": result["extraction"]["parameters"],
                      "derived_sha256": result["extraction"]["derived_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
