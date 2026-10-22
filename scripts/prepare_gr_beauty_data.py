"""Acquire and normalize GRID's published P5 Amazon-2014 Beauty data on CPU.

No TensorFlow is required. Existing GRID TFRecords can be passed with --source;
--download fetches just the Beauty ZIP entries using verified HTTP ranges.
Downloaded records and generated JSON belong outside the source checkout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
import tempfile
import urllib.request
import zipfile
import zlib


SOURCE_URL = "https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view"
DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz&export=download&confirm=t"
)
GRID_COMMIT = "2fe3475b2d369580234093f35d52b1a2f54d0472"
ARCHIVE_SIZE = 4_753_440_545
TAIL_SIZE = 262_144
TAIL_SHA256 = "76a79ca687d45121fa8735a8bccb0b313773bde193c6362f40d032b5751068f5"
NORMALIZED_SHA256 = {
    "catalog.json": "0336747ab3d50741fde2bb17f734b9e393c418bc54f818ab714977bacd2ae767",
    "sequences.json": "f197aab8b2e8a9d86bef04a4a552b112ce0c8d7be831dff6281e416783d2c645",
}
ENTRY_PATTERN = re.compile(
    r"amazon_data/beauty/(?:(?:training|evaluation|testing)/partition_\d+|items/data_\d+)\.tfrecord\.gz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, position
    raise ValueError("invalid protobuf varint")


def _fields(data: bytes):
    position = 0
    while position < len(data):
        key, position = _varint(data, position)
        tag, wire = key >> 3, key & 7
        if tag == 0:
            raise ValueError("invalid protobuf field number")
        if wire == 0:
            value, position = _varint(data, position)
        else:
            if wire == 2:
                length, position = _varint(data, position)
            elif wire in (1, 5):
                length = 8 if wire == 1 else 4
            else:
                raise ValueError("unsupported protobuf wire type")
            if position + length > len(data):
                raise ValueError("truncated protobuf field")
            value = data[position:position + length]
            position += length
        yield tag, wire, value


def decode_example(data: bytes) -> dict[str, list]:
    """Read only the sequence/identity/text fields of tensorflow.train.Example.

    Source precomputed float embeddings are deliberately ignored: their encoder
    provenance is unspecified and they are not our Sentence-T5 baseline.
    """
    wanted = {"id", "user_id", "sequence_data", "text"}
    top = list(_fields(data))
    if len(top) != 1 or top[0][:2] != (1, 2):
        raise ValueError("expected tf.train.Example.features")
    result = {}
    for tag, wire, entry in _fields(top[0][2]):
        if (tag, wire) != (1, 2):
            raise ValueError("invalid Features map entry")
        parts = {number: value for number, _, value in _fields(entry)}
        name = parts[1].decode("utf-8")
        if name not in wanted:
            continue
        if name in result:
            raise ValueError("duplicate feature key")
        typed = list(_fields(parts[2]))
        if len(typed) != 1 or typed[0][1] != 2:
            raise ValueError("expected one bytes_list or int64_list")
        kind, _, content = typed[0]
        values = []
        for number, wire, value in _fields(content):
            if number != 1:
                raise ValueError("unexpected list field")
            if kind == 1 and wire == 2:
                values.append(value)
            elif kind == 3 and wire == 0:
                values.append(value)
            elif kind == 3 and wire == 2:
                position = 0
                while position < len(value):
                    item, position = _varint(value, position)
                    values.append(item)
            else:
                raise ValueError("unsupported feature list type")
        result[name] = values
    return result


def read_examples(path: Path):
    with gzip.open(path, "rb") as stream:
        while header := stream.read(12):
            if len(header) != 12:
                raise ValueError("truncated TFRecord header")
            length = struct.unpack("<Q", header[:8])[0]
            if length > 128 * 1024 * 1024:
                raise ValueError("unexpectedly large TFRecord")
            record = stream.read(length)
            if len(record) != length or len(stream.read(4)) != 4:
                raise ValueError("truncated TFRecord payload")
            yield decode_example(record)


def merge_splits(training: dict, evaluation: dict, testing: dict) -> list[dict]:
    if training.keys() != evaluation.keys() or training.keys() != testing.keys():
        raise ValueError("train/validation/test user sets differ")
    records = []
    for user_id in sorted(training):
        train, val, test = training[user_id], evaluation[user_id], testing[user_id]
        if not train or len(val) < 2 or len(test) < 2:
            raise ValueError("empty history or missing held-out target")
        # GRID retains full training histories, but clips validation/testing rows
        # to twenty context items followed by one target. Retain ALL train items.
        if train[-20:] != val[:-1] or (train + [val[-1]])[-20:] != test[:-1]:
            raise ValueError(f"held-out context alignment failed for user {user_id}")
        records.append({"user_id": user_id, "train": train,
                        "validation": val[-1], "test": test[-1]})
    return records


def normalize(source: Path, output: Path, *, expected_counts: bool = True) -> dict:
    splits, source_files, split_summary = {}, [], {}
    for split in ("training", "evaluation", "testing"):
        users = {}
        paths = sorted((source / split).glob("*.tfrecord.gz"))
        if not paths:
            raise ValueError(f"missing {split} TFRecords")
        for path in paths:
            source_files.append(path)
            for row in read_examples(path):
                if len(row["user_id"]) != 1:
                    raise ValueError("expected one user ID per row")
                uid = row["user_id"][0]
                if uid in users:
                    raise ValueError("duplicate user in source split")
                users[uid] = row["sequence_data"]
        splits[split] = users
        lengths = [len(sequence) for sequence in users.values()]
        split_summary[split] = {"users": len(users), "events": sum(lengths),
                                "min_sequence_length": min(lengths),
                                "max_sequence_length": max(lengths)}
    sequences = merge_splits(splits["training"], splits["evaluation"], splits["testing"])
    catalog = {}
    for path in sorted((source / "items").glob("*.tfrecord.gz")):
        source_files.append(path)
        for row in read_examples(path):
            if len(row["id"]) != 1 or len(row["text"]) != 1:
                raise ValueError("expected one item ID and text per catalog row")
            item_id = row["id"][0]
            if item_id in catalog:
                raise ValueError("duplicate catalog item")
            catalog[item_id] = {"item_id": item_id, "text": row["text"][0].decode("utf-8")}
    used = {item for user in sequences
            for item in user["train"] + [user["validation"], user["test"]]}
    if used != catalog.keys() or set(catalog) != set(range(len(catalog))):
        raise ValueError("catalog must be complete and use contiguous zero-based item IDs")
    total_events = sum(len(user["train"]) + 2 for user in sequences)
    if expected_counts and (len(catalog), len(sequences), total_events) != (12101, 22363, 198502):
        raise ValueError("source differs from canonical P5 Amazon-2014 Beauty counts")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("catalog.json", [catalog[key] for key in sorted(catalog)]),
                        ("sequences.json", sequences)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
                                   encoding="utf-8")
        if expected_counts and sha256(output / name) != NORMALIZED_SHA256[name]:
            raise ValueError(f"normalized {name} differs from audited GRID Beauty contents")
    manifest = {
        "source": "GRID public P5 Amazon2014 Beauty archive", "source_url": SOURCE_URL,
        "grid_commit": GRID_COMMIT, "users": len(sequences), "unique_items": len(catalog),
        "all_events": total_events, "item_id_min": min(catalog), "item_id_max": max(catalog),
        "source_splits": split_summary, "source_embeddings_reused": False,
        "split": "full train history, penultimate validation, final test; supplied last20 contexts verified",
        "normalization": "exact UTF8 item text and original zero-based IDs; no train truncation/filtering",
        "files": {name: {"sha256": sha256(output / name), "size_bytes": (output / name).stat().st_size}
                  for name in ("catalog.json", "sequences.json")},
        "source_files": [{"path": str(path.relative_to(source)), "size_bytes": path.stat().st_size,
                          "sha256": sha256(path)} for path in source_files],
    }
    (output / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _range(start: int, end: int) -> bytes:
    request = urllib.request.Request(DOWNLOAD_URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if (response.status != 206 or
                response.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_SIZE}"):
            raise ValueError("source server did not honor the exact archive range")
        data = response.read()
    if len(data) != end - start + 1:
        raise ValueError("truncated archive range")
    return data


def download(destination: Path, workers: int = 8) -> Path:
    """Fetch exact published ZIP entries, checking pinned directory + CRC32."""
    tail = _range(ARCHIVE_SIZE - TAIL_SIZE, ARCHIVE_SIZE - 1)
    if hashlib.sha256(tail).hexdigest() != TAIL_SHA256:
        raise ValueError("GRID data archive directory changed; audit before repinning")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as sparse:
        sparse.truncate(ARCHIVE_SIZE)
        sparse.seek(ARCHIVE_SIZE - TAIL_SIZE)
        sparse.write(tail)
        with zipfile.ZipFile(sparse) as archive:
            entries = [info for info in archive.infolist() if ENTRY_PATTERN.fullmatch(info.filename)]
        if len(entries) != 549:
            raise ValueError("unexpected Beauty source entry count")

    def fetch(info):
        path = destination / info.filename
        if path.is_file() and path.stat().st_size == info.file_size:
            if zlib.crc32(path.read_bytes()) == info.CRC:
                return
        header = _range(info.header_offset, info.header_offset + 29)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("invalid ZIP entry header")
        name_length, extra_length = struct.unpack("<HH", header[26:30])
        start = info.header_offset + 30 + name_length + extra_length
        packed = _range(start, start + info.compress_size - 1)
        if info.compress_type != 8:
            raise ValueError("unexpected source ZIP compression method")
        data = zlib.decompress(packed, -15)
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise ValueError("source ZIP entry length or checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(fetch, entries))
    return destination / "amazon_data" / "beauty"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="existing beauty directory of GRID TFRecords")
    parser.add_argument("--download", type=Path, help="download exact Beauty entries into this scratch directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.source is None) == (args.download is None):
        parser.error("provide exactly one of --source and --download")
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    source = args.source if args.source is not None else download(args.download, args.workers)
    manifest = normalize(source, args.output)
    print(json.dumps({key: manifest[key] for key in ("users", "unique_items", "all_events", "files")}, indent=2))


if __name__ == "__main__":
    main()
