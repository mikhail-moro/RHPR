"""Catalogue-only structured features and PyTorch field reconstruction heads."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FIELD_NAMES = ("brand", "categories", "price")


def prepare_fields(texts, schema=None):
    """Reuse conservative published-text parsing; never infer missing metadata.

    Fitting uses the static catalogue, including held-out items, as does the
    original tokenizer. A supplied schema is applied without re-fitting it.
    """
    from rhpr.gr.study_index import extract_features

    rows = [{name: value for name, value, status in extract_features(text)} for text in texts]
    if not rows:
        raise ValueError("field encoder requires a nonempty catalogue")
    brands = [row["brand"].strip() if row["brand"] else None for row in rows]
    categories = [sorted(set(row["categories"] or [])) for row in rows]
    prices = np.array([row["price"] if row["price"] is not None else np.nan for row in rows])
    price_observed = np.isfinite(prices)
    if schema is None:
        if not price_observed.any() or not any(brands) or not any(categories):
            raise ValueError("each selected field needs at least one observed catalogue value")
        scale = float(prices[price_observed].std(ddof=0))
        schema = {"version": 1, "fields": list(FIELD_NAMES),
            "brand_vocabulary": sorted(set(filter(None, brands))),
            "category_vocabulary": sorted({value for row in categories for value in row}),
            "price_mean": float(prices[price_observed].mean()), "price_scale": scale if scale > 0 else 1.0,
            "price_ddof": 0, "constant_price": scale == 0,
            "fit_catalogue_text_sha256": hashlib.sha256(json.dumps(list(texts), ensure_ascii=False).encode()).hexdigest(),
            "provenance": "parsed_from_encoder_text_v1; no title, metadata imputation or interactions"}
    if schema["fields"] != list(FIELD_NAMES) or schema["version"] != 1:
        raise ValueError("unsupported field schema")
    if not np.isfinite([schema["price_mean"], schema["price_scale"]]).all() or schema["price_scale"] <= 0:
        raise ValueError("invalid saved price standardization")
    brand_ids = {value: i + 1 for i, value in enumerate(schema["brand_vocabulary"])}
    category_ids = {value: i for i, value in enumerate(schema["category_vocabulary"])}
    # New nonmissing values cannot silently become a known class or be dropped.
    if any(value and value not in brand_ids for value in brands):
        raise ValueError("brand outside fitted catalogue vocabulary")
    if any(value not in category_ids for row in categories for value in row):
        raise ValueError("category outside fitted catalogue vocabulary")
    multi_hot = np.zeros((len(rows), len(category_ids)), dtype=np.float32)
    for i, row in enumerate(categories):
        for value in row:
            multi_hot[i, category_ids[value]] = 1
    arrays = {"brand": np.array([brand_ids.get(value, 0) for value in brands], dtype=np.int64),
        "categories": multi_hot, "categories_observed": multi_hot.sum(1) > 0,
        "price": np.where(price_observed, (prices - schema["price_mean"]) / schema["price_scale"], 0).astype(np.float32)[:, None],
        "price_observed": price_observed}
    return schema, arrays


def tensor_fields(arrays, indices, device):
    return {name: torch.as_tensor(values[indices], device=device) for name, values in arrays.items()}


class FieldEncoder(nn.Module):
    def __init__(self, schema, field_dim=64):
        super().__init__()
        if type(field_dim) is not int or field_dim < 1:
            raise ValueError("field_dim must be a positive integer")
        self.output_dim = 3 * field_dim
        self.brand = nn.Embedding(len(schema["brand_vocabulary"]) + 1, field_dim, padding_idx=0)
        self.categories = nn.Embedding(len(schema["category_vocabulary"]), field_dim)
        self.projections = nn.ModuleDict({
            "brand": nn.Linear(field_dim + 1, field_dim),
            "categories": nn.Linear(field_dim + 1, field_dim),
            "price": nn.Linear(2, field_dim)})

    def forward(self, fields):
        brand = self.brand(fields["brand"])
        labels = fields["categories"]
        # Exactly mean-pooling selected embeddings; duplicate labels were removed.
        categories = (labels @ self.categories.weight) / labels.sum(-1, keepdim=True).clamp_min(1)
        values = {"brand": brand, "categories": categories, "price": fields["price"]}
        observed = {"brand": fields["brand"] > 0, "categories": fields["categories_observed"],
                    "price": fields["price_observed"]}
        return torch.cat([self.projections[name](torch.cat([
            values[name], observed[name][:, None].to(values[name].dtype)], dim=-1))
            for name in FIELD_NAMES], dim=-1)


class FieldHeads(nn.Module):
    def __init__(self, schema, input_dim):
        super().__init__()
        self.heads = nn.ModuleDict({"brand": nn.Linear(input_dim, len(schema["brand_vocabulary"])),
            "categories": nn.Linear(input_dim, len(schema["category_vocabulary"])),
            "price": nn.Linear(input_dim, 1)})

    def forward(self, decoded):
        return {name: head(decoded) for name, head in self.heads.items()}

    def losses(self, predictions, fields):
        masks = {"brand": fields["brand"] > 0, "categories": fields["categories_observed"],
                 "price": fields["price_observed"]}
        losses, counts = {}, {}
        for name in FIELD_NAMES:
            mask = masks[name]
            counts[name] = int(mask.sum())
            if not counts[name]:
                losses[name] = predictions[name].sum() * 0
            elif name == "brand":
                losses[name] = F.cross_entropy(predictions[name][mask], fields[name][mask] - 1)
            elif name == "categories":
                losses[name] = F.binary_cross_entropy_with_logits(predictions[name][mask], fields[name][mask])
            else:
                losses[name] = F.huber_loss(predictions[name][mask], fields[name][mask], delta=1.0)
        return losses, counts
