"""Finite, auditable next-item examples for the GR baseline.

Catalogue rows are mapped to zero-based internal item IDs. Source IDs remain in
the prepared data; neither item IDs nor user IDs are sent to the text encoder.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class UserSequence:
    user_id: int
    train: tuple[int, ...]
    validation: int
    test: int


def load_beauty(directory: Path, expected: dict | None = None):
    """Load the lossless normalized GRID data, checking the declared identity."""
    expected = expected or {}
    for name, digest in expected.get("sha256", {}).items():
        if Path(name).name != name or file_sha256(directory / name) != digest:
            raise ValueError(f"prepared data digest mismatch: {name}")
    catalog = json.loads((directory / "catalog.json").read_text())
    sequences = json.loads((directory / "sequences.json").read_text())
    source_ids = [row["item_id"] for row in catalog]
    if (not source_ids or any(type(value) is not int for value in source_ids)
            or len(set(source_ids)) != len(source_ids)):
        raise ValueError("catalogue item IDs must be unique integers")
    item_map = {value: index for index, value in enumerate(source_ids)}
    texts = [row["text"] for row in catalog]
    if any(not isinstance(value, str) or not value.strip() for value in texts):
        raise ValueError("every catalogue item must have nonempty text")
    users = []
    for row in sequences:
        values = [*row["train"], row["validation"], row["test"]]
        if type(row["user_id"]) is not int or any(type(value) is not int for value in values):
            raise ValueError("sequence IDs must be integers")
        if len(row["train"]) < 2 or any(value not in item_map for value in values):
            raise ValueError("sequence has insufficient training data or unknown items")
        users.append(UserSequence(row["user_id"], tuple(item_map[x] for x in row["train"]),
                                  item_map[row["validation"]], item_map[row["test"]]))
    if not users or len({user.user_id for user in users}) != len(users):
        raise ValueError("user IDs must be unique")
    counts = {"items": len(catalog), "users": len(users),
              "train_events": sum(len(user.train) for user in users),
              "train_examples": sum(len(user.train) - 1 for user in users)}
    for key, actual in counts.items():
        if key in expected and expected[key] != actual:
            raise ValueError(f"prepared data count mismatch: {key}")
    return texts, users, source_ids, counts


def build_examples(users: list[UserSequence], history_items: int, split: str):
    """One supervised example for every noninitial train event, or held-out user.

    Training covers every available next-item target once per epoch. Histories
    contain at most history_items observations before that target. Test history
    includes the already observed validation item, with no test label in input.
    """
    import numpy as np

    if type(history_items) is not int or history_items < 1:
        raise ValueError("history_items must be a positive integer")
    if split not in {"train", "validation", "test"}:
        raise ValueError("unknown split")
    count = sum(len(u.train) - 1 if split == "train" else 1 for u in users)
    contexts = np.full((count, history_items), -1, dtype=np.int64)
    targets = np.empty(count, dtype=np.int64)
    user_ids = np.empty(count, dtype=np.int64)
    index = 0
    for user in users:
        history = user.train if split != "test" else (*user.train, user.validation)
        positions = range(1, len(history)) if split == "train" else [len(history)]
        for position in positions:
            prefix = history[max(0, position - history_items):position]
            contexts[index, -len(prefix):] = prefix
            targets[index] = (history[position] if split == "train" else
                              user.validation if split == "validation" else user.test)
            user_ids[index] = user.user_id
            index += 1
    return contexts, user_ids, targets
