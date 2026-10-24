"""Data-only index evidence. Behavioural statistics never read held-out labels."""
from __future__ import annotations

from collections import Counter, defaultdict
import ast
import hashlib
import json
import math
import re

import numpy as np

from rhpr.gr.study_store import prefix_key


def entropy(counts):
    total = sum(counts)
    return -sum((n / total) * math.log2(n / total) for n in counts if n) if total else None


def extract_features(text):
    """Conservative extraction from published encoder text, never source truth.

    Preserve raw text separately. Missing/ambiguous values remain explicit.
    """
    match = re.fullmatch(r"Title: (.*); Brand: (.*); Categories: (.*); Price: (.*);\s*", text, re.S)
    if not match:
        return [(name, None, "unparsed") for name in ("title", "brand", "categories", "price")]
    title, brand, categories, price = match.groups()
    try:
        category = ast.literal_eval(categories)
        if not isinstance(category, list) or not all(isinstance(x, str) for x in category):
            raise ValueError()
    except (ValueError, SyntaxError):
        category = None
    try:
        number = float(price)
        if not math.isfinite(number):
            raise ValueError()
    except ValueError:
        number = None
    return [("title", title, "parsed"), ("brand", None if brand == "Unknown" else brand,
             "missing" if brand == "Unknown" else "parsed"),
            ("categories", category, "parsed" if category is not None else "unparsed"),
            ("price", number, "parsed" if number is not None else "unparsed")]


def prepare_shared(store, texts, source_ids, users, *, context_items, panel_size, panel_seed):
    """Full events, actual evaluation windows, sparse train graph, shared panel."""
    frequency, user_frequency = Counter(), Counter()
    transition, co_users = Counter(), Counter()
    for user in users:
        frequency.update(user.train)
        user_frequency.update(set(user.train))
        transition.update(zip(user.train, user.train[1:]))
        unique = sorted(set(user.train))
        co_users.update((a, b) for a in unique for b in unique if a != b)
    store.write("items", [{"item_id": i, "source_id": original, "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "train_events": frequency[i],
        "train_users": user_frequency[i]} for i, (text, original) in enumerate(zip(texts, source_ids, strict=True))])
    store.write("item_features", [{"item_id": i, "name": name, "value_json": json.dumps(value, ensure_ascii=False),
        "provenance": "parsed_from_encoder_text_v1", "status": status}
        for i, text in enumerate(texts) for name, value, status in extract_features(text)])
    events, cases, histories = [], [], []
    for user in users:
        for position, item in enumerate((*user.train, user.validation, user.test)):
            events.append({"user_id": user.user_id, "position": position, "item_id": item,
                "split": "train" if position < len(user.train) else "validation" if position == len(user.train) else "test",
                "timestamp": None})
        for split, history, target in (("validation", user.train, user.validation),
                ("test", (*user.train, user.validation), user.test)):
            case_id = f"{split}:{user.user_id}:{len(history)}"
            positions = [p for p, item in enumerate(history) if item == target]
            cases.append({"case_id": case_id, "user_id": user.user_id, "split": split,
                "target_position": len(history), "target_item": target, "history": list(history),
                "visible_history": list(history[-context_items:]), "target_repeats": len(positions),
                "target_last_lag": len(history) - positions[-1] if positions else None, "exact_panel": False,
                "panel_stratum": None, "panel_probability": None})
            start = max(0, len(history) - context_items)
            histories.extend({"case_id": case_id, "position": p, "item_id": item,
                "visible": p >= start, "input_position": p - start if p >= start else None}
                for p, item in enumerate(history))
    # Outcome-blind stratification on TRAIN target popularity and context length.
    # Test labels do not select the validation panel. Each split gets its own
    # predeclared panel using only context length and TRAIN target popularity.
    rng = np.random.default_rng(panel_seed)
    for split in ("validation", "test"):
        groups = defaultdict(list)
        for row in cases:
            if row["split"] == split:
                key = (min(3, int(math.log2(len(row["visible_history"])))),
                       min(4, int(math.log2(1 + frequency[row["target_item"]]))))
                row["panel_stratum"] = prefix_key(key)
                groups[key].append(row)
        for group in groups.values():
            rng.shuffle(group)
        selected = 0
        while groups and selected < panel_size:
            for key in sorted(list(groups)):
                if selected == panel_size:
                    break
                groups[key].pop()["exact_panel"] = True
                selected += 1
                if not groups[key]:
                    del groups[key]
        stratum_counts, selected_counts = Counter(), Counter()
        for row in cases:
            if row["split"] == split:
                stratum_counts[row["panel_stratum"]] += 1
                selected_counts[row["panel_stratum"]] += int(row["exact_panel"])
        for row in cases:
            if row["split"] == split:
                row["panel_probability"] = selected_counts[row["panel_stratum"]] / stratum_counts[row["panel_stratum"]]
    store.write("events", events)
    store.write("cases", cases)
    store.write("case_history", histories)
    store.write("train_edges", [{"source_item": a, "target_item": b, "kind": kind, "count": count,
        "source_support": support[a], "target_support": support[b]}
        for kind, edges, support in (("next_item", transition, frequency), ("co_user", co_users, user_frequency))
        for (a, b), count in sorted(edges.items())])
    return cases


def window_manifest(users, *, seed, epoch, source_size, maximum, context_items):
    """Author TIGER/GRID contiguous-window sampling, expressed in raw item IDs.

    Source draws with replacement then enumerates membership: duplicate draws
    are emitted once. The manifest isolates sampling from quantization/dropout.
    end is EXCLUSIVE and includes the target at end-1.
    """
    import torch
    from rhpr.gr.genpas_epochs import epoch_order
    rng = torch.Generator().manual_seed(seed + epoch - 1)
    order = epoch_order(len(users), seed, epoch - 1)
    result = []
    for group, begin in enumerate(range(0, len(order), source_size)):
        subset = [users[i] for i in order[begin:begin + source_size]]
        totals = [len(u.train) * (len(u.train) - 1) // 2 for u in subset]
        total = sum(totals)
        selected = (torch.randint(total, (maximum,), generator=rng).unique(sorted=True).tolist()
                    if total > maximum else range(total))
        user_index, base = 0, 0
        for number in selected:
            while number >= base + totals[user_index]:
                base += totals[user_index]
                user_index += 1
            local = number - base
            end = (1 + math.isqrt(1 + 8 * local)) // 2 + 1
            start = local - (end - 1) * (end - 2) // 2
            user = subset[user_index]
            result.append({"epoch": epoch, "ordinal": len(result), "source_group": group,
                "user_id": user.user_id, "start": start, "end": end,
                "visible_start": max(start, end - 1 - context_items), "target_item": user.train[end - 1]})
    return result


def record_index(store, run_id, index, users, cases, values, centers):
    """Index geometry plus catalogue/train statistics, including collision leaves."""
    from sklearn.metrics import pairwise_distances
    store.write("item_codes", [{"run_id": run_id, "item_id": int(i), "sid": row.tolist()}
                for i, row in zip(index.item_ids, index.sids, strict=True)])
    residual = values.astype(np.float32).copy()
    for level, center in enumerate(centers):
        norms = np.linalg.norm(residual, axis=1)
        residual /= np.maximum(norms[:, None], 1e-12)
        distances = pairwise_distances(residual, center, metric="euclidean")
        order = np.argsort(distances, axis=1, kind="stable")
        assigned = index.semantic_codes[:, level]
        alternative = np.where(order[:, 0] == assigned, order[:, 1], order[:, 0])
        residual -= center[assigned]
        after = np.linalg.norm(residual, axis=1)
        store.write("geometry", [{"run_id": run_id, "item_id": i, "level": level,
            "token": int(assigned[i]), "nearest_token": int(order[i, 0]), "alternative_token": int(alternative[i]),
            "assigned_distance": float(distances[i, assigned[i]]), "alternative_distance": float(distances[i, alternative[i]]),
            "distance_margin": float(distances[i, alternative[i]] - distances[i, assigned[i]]),
            "residual_norm_before": float(norms[i]), "residual_norm_after": float(after[i])}
            for i in range(index.num_items)])
    frequency = Counter(item for u in users for item in u.train)
    targets = Counter(item for u in users for item in u.train[1:])
    membership = defaultdict(set)
    for user in users:
        for item in set(user.train):
            for depth in range(index.sid_length + 1):
                membership[tuple(index.sids[item, :depth])].add(user.user_id)
    groups = defaultdict(list)
    for i, sid in enumerate(index.sids):
        for depth in range(index.sid_length + 1):
            groups[tuple(sid[:depth])].append(i)
    rows = []
    for prefix, items in sorted(groups.items()):
        depth = len(prefix)
        children = Counter()
        child_tokens = set()
        if depth < index.sid_length:
            for i in items:
                token = int(index.sids[i, depth])
                children[token] += targets[i]
                child_tokens.add(token)
        total = sum(children.values())
        rows.append({"run_id": run_id, "prefix": prefix_key(prefix),
            "parent": prefix_key(prefix[:-1]) if prefix else None, "level": depth - 1,
            "token": int(prefix[-1]) if prefix else None, "is_suffix": depth == index.sid_length,
            "items": len(items), "train_events": sum(frequency[i] for i in items),
            "train_users": len(membership[prefix]), "child_count": len(child_tokens),
            "target_events": sum(targets[i] for i in items), "child_entropy": entropy(children.values()),
            "child_max_share": max(children.values()) / total if total else None})
    store.write("prefixes", rows)
    store.write("token_usage", [{"run_id": run_id, "level": level, "token": int(token),
        "items": int((index.sids[:, level] == token).sum()),
        "parent_count": len({tuple(sid[:level]) for sid in index.sids if sid[level] == token}),
        "train_events": sum(frequency[i] for i in range(index.num_items) if index.sids[i, level] == token)}
        for level in range(index.sid_length) for token in np.unique(index.sids[:, level])])
    for begin in range(0, len(cases), 2048):
        rows = []
        for case in cases[begin:begin + 2048]:
            history, target = case["visible_history"], index.sids[case["target_item"]]
            for level in range(index.sid_length):
                prefixes = [tuple(index.sids[item, :level + 1]) for item in history]
                matching = [p for p, prefix in enumerate(prefixes) if prefix == tuple(target[:level + 1])]
                rows.append({"run_id": run_id, "case_id": case["case_id"], "level": level,
                    "distinct_prefixes": len(set(prefixes)), "prefix_entropy": entropy(Counter(prefixes).values()),
                    "target_shared_events": len(matching), "target_shared_last_lag": len(history) - matching[-1] if matching else None})
        store.write("case_index", rows)


def compare_indices(store, indices):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    names = sorted(indices)
    for n, a in enumerate(names):
        for b in names[n + 1:]:
            rows = []
            for level in range(indices[a].sid_length):
                x, y = indices[a].sids, indices[b].sids
                px = np.unique(x[:, :level + 1], axis=0, return_inverse=True)[1]
                py = np.unique(y[:, :level + 1], axis=0, return_inverse=True)[1]
                rows.append({"run_a": a, "run_b": b, "level": level,
                    "prefix_ari": float(adjusted_rand_score(px, py)), "prefix_nmi": float(normalized_mutual_info_score(px, py)),
                    "token_ari": float(adjusted_rand_score(x[:, level], y[:, level])),
                    "token_nmi": float(normalized_mutual_info_score(x[:, level], y[:, level]))})
            store.write("index_comparisons", rows)
