"""Prepare pinned Flan-T5-XXL encoder-only BF16 weights using bounded HTTP ranges.

Never download decoder tensors or allocate the whole model. Conversion from
published FP32 to BF16 is explicit. Source headers/index and derived bytes are
hashed; shard-wide hashes are recorded but not claimed verified by subset reads.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import struct
import threading
import urllib.request

MODEL = "google/flan-t5-xxl"
REVISION = "ae7c9136adc7555eeccc78cdd960dfd60fb346ce"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def ranged(name, start, end, size):
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?rhpr_range={start}-{end}"
    error = None
    for _ in range(4):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise ValueError("Published model server did not honor the pinned range")
                data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise ValueError("Truncated model tensor range")
            return data
        except Exception as caught:
            error = caught
    raise error


def prepare(output, workers=4):
    import torch
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    # API metadata pins immutable file sizes and full-shard publisher hashes.
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true") as response:
        metadata = json.load(response)
    files = {row["rfilename"]: row for row in metadata["siblings"]}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json", "tokenizer.json",
                 "special_tokens_map.json", "spiece.model", "README.md"):
        if not (output / name).exists():
            with urllib.request.urlopen(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}") as response:
                (output / name).write_bytes(response.read())
    if not (output / "LICENSE").exists():
        with urllib.request.urlopen("https://www.apache.org/licenses/LICENSE-2.0.txt") as response:
            (output / "LICENSE").write_bytes(response.read())
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    expected = {name: shard for name, shard in index.items() if name.startswith(("encoder.", "shared."))}
    headers = {}
    for shard in sorted(set(expected.values())):
        size = files[shard]["size"]
        length = struct.unpack("<Q", ranged(shard, 0, 7, size))[0]
        raw = ranged(shard, 8, 8 + length - 1, size)
        headers[shard] = (8 + length, json.loads(raw), size, hashlib.sha256(raw).hexdigest())
    selected = {}
    aliases = {}
    for name, shard in expected.items():
        body, header, size, _ = headers[shard]
        if name not in header:
            if name == "encoder.embed_tokens.weight" and "shared.weight" in expected:
                aliases[name] = "shared.weight"
                continue
            raise ValueError(f"Missing real encoder tensor: {name}")
        tensor = header[name]
        if tensor["dtype"] != "F32" or tensor["data_offsets"][1] - tensor["data_offsets"][0] != math.prod(tensor["shape"]) * 4:
            raise ValueError("Unexpected source tensor representation")
        selected[name] = dict(tensor, shard=shard, body=body, source_size=size)
    packed, offset, jobs = {"__metadata__": {"format": "pt", **aliases}}, 0, []
    for name, tensor in sorted(selected.items()):
        count = math.prod(tensor["shape"])
        packed[name] = {"dtype": "BF16", "shape": tensor["shape"], "data_offsets": [offset, offset + count * 2]}
        # Multiple aligned chunks keep RAM bounded even for the shared embedding.
        for begin in range(0, count, 4 << 20):
            length = min(4 << 20, count - begin)
            jobs.append((name, begin, length, offset + begin * 2))
        offset += count * 2
    encoded = json.dumps(packed, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    prefix = struct.pack("<Q", len(encoded)) + encoded
    target = output / "model.safetensors"
    partial = target.with_suffix(".partial")
    state_path = output / "conversion-progress.json"
    identity = hashlib.sha256(prefix).hexdigest()
    if target.exists():
        return json.loads((output / "encoder-provenance.json").read_text())
    completed = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["header_sha256"] != identity:
            raise ValueError("Partial encoder belongs to another model")
        completed = state["completed"]
    else:
        with partial.open("wb") as stream:
            stream.write(prefix)
            stream.truncate(len(prefix) + offset)
    lock = threading.Lock()

    def fetch(job):
        name, begin, length, destination = job
        key = name + ":" + str(begin)
        if key in completed:
            return
        tensor = selected[name]
        start = tensor["body"] + tensor["data_offsets"][0] + begin * 4
        raw = ranged(tensor["shard"], start, start + length * 4 - 1, tensor["source_size"])
        values = torch.frombuffer(bytearray(raw), dtype=torch.float32).to(torch.bfloat16)
        converted = values.view(torch.uint16).numpy().tobytes()
        with partial.open("r+b") as stream:
            stream.seek(len(prefix) + destination)
            stream.write(converted)
        with lock:
            completed[key] = {"fp32_sha256": hashlib.sha256(raw).hexdigest(),
                              "bf16_sha256": hashlib.sha256(converted).hexdigest()}
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"header_sha256": identity, "completed": completed}))
            temporary.replace(state_path)
            if len(completed) % 32 == 0 or len(completed) == len(jobs):
                print(json.dumps({"phase": "xxl_encoder_conversion", "chunks": len(completed),
                                  "total_chunks": len(jobs), "bf16_bytes": offset}), flush=True)
    print(json.dumps({"phase": "xxl_encoder_start", "tensor_parameters": offset // 2,
                      "bf16_bytes": offset, "chunks": len(jobs)}), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, jobs))
    partial.replace(target)
    report = {"model": MODEL, "revision": REVISION, "dtype": "bfloat16",
        "source_dtype": "float32", "conversion": "PyTorch round-to-nearest BF16",
        "parameters": offset // 2, "encoder_only": True,
        "source_headers": {name: {"sha256": value[3], "publisher_shard": files[name]} for name, value in headers.items()},
        "source_whole_shard_hashes_verified": False, "range_hashes": completed,
        "files": [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": digest(p)} for p in
            sorted(output.iterdir()) if p.name not in {"encoder-provenance.json", "conversion-progress.json"}],
        "model_sha256": digest(target)}
    (output / "encoder-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    value = prepare(args.output, args.workers)
    print(json.dumps({name: value[name] for name in ("model", "revision", "parameters", "model_sha256")}))
