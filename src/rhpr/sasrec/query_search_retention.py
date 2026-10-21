"""Bound the disk occupied by complete optimizer/RNG trial checkpoints."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
import errno
import math
import os
from pathlib import Path
import stat
from typing import Any


_COMPLETED_STATUSES = frozenset(("training_completed", "success", "full_transfer_failed"))


def prune_trial_resumes(output_dir: Path,
                        trials: Mapping[str, Mapping[str, dict[str, Any]]],
                        resume_records: Sequence[dict[str, Any]],
                        limit: int) -> list[dict[str, Any]]:
    """Retain the best finite short-validation resumes of completed fresh trials.

    Call between trials, when no training job owns an active resume. Scores are
    ``best_validation.ndcg10``; ties keep the nested report's insertion order.
    Historical ``resumable_checkpoint`` metadata is preserved on every result,
    and ``resume_checkpoint_retained`` reports whether its verified resume is
    kept. Returned manifest records retain their original order.

    Only exact ``checkpoints/{family}-{trial}-resume.pt`` paths registered in
    both the manifest and the owning result can be deleted. Metadata must mark
    them as optimizer checkpoints and match their current file size. Best
    weights, missing files, symlinks and unrecognized paths are never deleted.
    Directory-relative operations prevent a symlinked checkpoint directory
    from redirecting deletion outside the explicit output directory.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("resume retention limit must be a nonnegative integer")

    owners: dict[tuple[str, str], dict[str, Any]] = {}
    protected_paths = set()
    path_owners: dict[str, list[tuple[str, str]]] = {}
    for family, family_trials in trials.items():
        for name, result in family_trials.items():
            owners[family, name] = result
            path_owners.setdefault(f"checkpoints/{family}-{name}-resume.pt", []).append((family, name))
            result["resume_checkpoint_retained"] = False
            best_path = result.get("best_checkpoint", {}).get("path")
            if isinstance(best_path, str):
                protected_paths.add(os.path.normpath(best_path))

    with ExitStack() as stack:
        try:
            root_fd = os.open(Path(output_dir).resolve(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, root_fd)
            directory_fd = os.open("checkpoints", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                   dir_fd=root_fd)
            stack.callback(os.close, directory_fd)
        except (FileNotFoundError, NotADirectoryError):
            return []
        except OSError as error:
            # O_NOFOLLOW rejects a symlink rather than opening its target.
            if error.errno == errno.ELOOP:
                return []
            raise

        verified: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
        for record in resume_records:
            family = record.get("training_model")
            trial = record.get("trial")
            name = trial.get("name") if isinstance(trial, Mapping) else None
            if not isinstance(family, str) or not isinstance(name, str):
                continue
            key = family, name
            result = owners.get(key)
            if result is None:
                continue
            path = record.get("path")
            expected = f"checkpoints/{family}-{name}-resume.pt"
            if (not isinstance(path, str) or path != expected or path in protected_paths
                    or len(path_owners[expected]) != 1
                    or Path(path).parts != ("checkpoints", f"{family}-{name}-resume.pt")
                    or record.get("optimizer_state_saved") is not True):
                continue
            historical = result.get("resumable_checkpoint", {})
            if (historical.get("path") != path or historical.get("training_model") != family
                    or historical.get("trial", {}).get("name") != name
                    or historical.get("optimizer_state_saved") is not True
                    or historical.get("size_bytes") != record.get("size_bytes")):
                continue
            filename = Path(path).name
            if not _verified_file(directory_fd, filename, record):
                continue
            verified[key] = record, filename

        candidates = []
        for key, result in owners.items():
            score = result.get("best_validation", {}).get("ndcg10")
            if (key in verified and result.get("from_scratch") is True
                    and result.get("complete") is True
                    and result.get("status") in _COMPLETED_STATUSES
                    and not isinstance(score, bool) and isinstance(score, (int, float))
                    and math.isfinite(score)):
                candidates.append((key, score))
        # Python's stable sort preserves report order for equal scores.
        candidates.sort(key=lambda row: -row[1])
        keep = {key for key, _score in candidates[:limit]}
        for key, (_record, filename) in verified.items():
            if key not in keep:
                try:
                    os.unlink(filename, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass

        retained_ids = set()
        for key in keep:
            record, filename = verified[key]
            if _verified_file(directory_fd, filename, record):
                owners[key]["resume_checkpoint_retained"] = True
                retained_ids.add(id(record))
        return [record for record in resume_records if id(record) in retained_ids]


def _verified_file(directory_fd: int, filename: str, record: Mapping[str, Any]) -> bool:
    expected_size = record.get("size_bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        return False
    try:
        current = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return stat.S_ISREG(current.st_mode) and current.st_size == expected_size
