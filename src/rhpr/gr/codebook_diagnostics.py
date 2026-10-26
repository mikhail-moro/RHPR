"""Full-catalogue RQ occupancy, geometry and unresolved SID collisions."""
import numpy as np


def semantic_statistics(codes, width):
    codes = np.asarray(codes)
    if codes.ndim != 2 or not len(codes) or codes.dtype.kind not in "iu" or width < 1:
        raise ValueError("expected nonempty integer semantic codes and positive width")
    if (codes < 0).any() or (codes >= width).any():
        raise ValueError("code outside codebook width")
    prefixes = []
    for level in range(codes.shape[1]):
        keys, sizes = np.unique(codes[:, :level + 1], axis=0, return_counts=True)
        hist_sizes, hist_counts = np.unique(sizes, return_counts=True)
        largest = np.argsort(-sizes, kind="stable")[:10]
        prefixes.append({"levels": level + 1, "unique_prefixes": len(sizes),
            "collision_groups": int((sizes > 1).sum()), "collided_items": int(sizes[sizes > 1].sum()),
            "collision_excess_items": len(codes) - len(sizes), "max_collision_group": int(sizes.max()),
            "groups_exceeding_suffix_width": int((sizes > width).sum()),
            "items_in_overflow_groups": int(sizes[sizes > width].sum()),
            "group_size_histogram": {str(int(s)): int(n) for s, n in zip(hist_sizes, hist_counts)},
            "largest_groups": [{"prefix": keys[i].tolist(), "items": int(sizes[i])} for i in largest]})
    return {"items": len(codes), "width": int(width), "semantic_levels": codes.shape[1],
        "prefixes": prefixes, "semantic_unique_ids": prefixes[-1]["unique_prefixes"],
        "max_collision_group": prefixes[-1]["max_collision_group"],
        "suffix_representable": prefixes[-1]["max_collision_group"] <= width,
        "collision_definition": "before suffix; intermediate-prefix groups are not full-SID failures"}


def diagnose_codebooks(codes, encoded, codebooks):
    codes, encoded, codebooks = np.asarray(codes), np.asarray(encoded), np.asarray(codebooks)
    if codebooks.ndim != 3 or encoded.shape != (len(codes), codebooks.shape[-1]):
        raise ValueError("codebooks/latent shapes disagree")
    if codes.shape[1] != len(codebooks) or not np.isfinite(encoded).all() or not np.isfinite(codebooks).all():
        raise ValueError("nonfinite or incompatible codebook inputs")
    width = codebooks.shape[1]
    summary = semantic_statistics(codes, width)
    codes = codes.astype(np.int64)
    # Float64 reductions avoid cancellation when comparing tiny residuals/norms.
    residual = encoded.astype(np.float64)
    quantized = np.zeros_like(residual)
    levels = []
    for level, book in enumerate(codebooks.astype(np.float64)):
        counts = np.bincount(codes[:, level], minlength=width)
        p = counts[counts > 0] / len(codes)
        entropy = float(-(p * np.log2(p)).sum())
        selected = book[codes[:, level]]
        error = residual - selected
        norms = np.linalg.norm(book, axis=1)
        pair_sq = np.maximum((book * book).sum(1)[:, None] + (book * book).sum(1)[None, :] - 2 * book @ book.T, 0)
        np.fill_diagonal(pair_sq, np.inf)
        levels.append({"level": level + 1, "counts": counts.tolist(),
            "used_codes": int((counts > 0).sum()), "dead_codes": int((counts == 0).sum()),
            "usage_fraction": float((counts > 0).mean()), "entropy_bits": entropy,
            "perplexity": float(2 ** entropy), "largest_code_fraction": float(counts.max() / len(codes)),
            "residual_mse_before": float(np.mean(residual ** 2)),
            "residual_mse_after": float(np.mean(error ** 2)),
            "centroid_norm_min": float(norms.min()), "centroid_norm_mean": float(norms.mean()),
            "centroid_norm_max": float(norms.max()), "centroid_mean_dim_variance": float(book.var(0).mean()),
            "distinct_centroids_exact": len(np.unique(book, axis=0)),
            "nearest_centroid_distance_mean": float(np.sqrt(pair_sq.min(1)).mean()) if width > 1 else None})
        quantized += selected
        residual = error
    summary.update(levels=levels, latent_norm_mean=float(np.linalg.norm(encoded, axis=1).mean()),
        latent_mean_dim_variance=float(encoded.astype(np.float64).var(0).mean()),
        quantized_mean_dim_variance=float(quantized.var(0).mean()),
        quantization_mse=float(np.mean(residual ** 2)))
    dimension = encoded.shape[1]
    zero_mse = float(np.mean(encoded.astype(np.float64) ** 2))
    mean_mse = summary["latent_mean_dim_variance"]
    summary.update(latent_dim=dimension,
        latent_norm_mean_over_sqrt_dim=summary["latent_norm_mean"] / float(np.sqrt(dimension)),
        quantized_norm_mean=float(np.linalg.norm(quantized, axis=1).mean()),
        quantized_norm_mean_over_sqrt_dim=float(np.linalg.norm(quantized, axis=1).mean() / np.sqrt(dimension)),
        latent_zero_predictor_mse=zero_mse,
        quantization_mse_over_zero_predictor_mse=summary["quantization_mse"] / zero_mse if zero_mse > 0 else None,
        quantization_mse_over_mean_predictor_mse=summary["quantization_mse"] / mean_mse if mean_mse > 0 else None)
    return summary
