"""Catalogue-only structured features and PyTorch field reconstruction heads."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FIELD_NAMES = ("brand", "categories", "price")


def prepare_fields(texts, schema=None, *, title_embeddings=None, title_observed=None):
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
    if schema["fields"] not in (list(FIELD_NAMES), [*FIELD_NAMES, "title"]) or schema["version"] != 1:
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
    if title_embeddings is not None:
        titles = np.asarray(title_embeddings, dtype=np.float32)
        if titles.ndim != 2 or titles.shape[0] != len(rows) or titles.shape[1] < 1 or not np.isfinite(titles).all():
            raise ValueError("title embeddings must be a finite catalogue-aligned matrix")
        observed = np.asarray(title_observed if title_observed is not None else
            [bool(row["title"] and row["title"].strip()) for row in rows], dtype=bool)
        if observed.shape != (len(rows),) or not observed.any():
            raise ValueError("title observation mask must match catalogue with observed titles")
        if "title_dim" in schema and schema["title_dim"] != titles.shape[1]:
            raise ValueError("title embedding dimension differs from saved schema")
        schema = {**schema, "fields": [*FIELD_NAMES, "title"], "title_dim": titles.shape[1],
            "title_embedding_values_sha256": hashlib.sha256(titles.tobytes()).hexdigest(),
            "provenance": "parsed_from_encoder_text_v1; frozen title embeddings; no metadata imputation or interactions"}
        arrays.update(title=np.where(observed[:, None], titles, 0), title_observed=observed)
    elif "title" in schema["fields"]:
        raise ValueError("saved title schema requires frozen title embeddings")
    return schema, arrays


def tensor_fields(arrays, indices, device):
    return {name: torch.as_tensor(values[indices], device=device) for name, values in arrays.items()}


class FieldEncoder(nn.Module):
    def __init__(self, schema, field_dim=64):
        super().__init__()
        if type(field_dim) is not int or field_dim < 1:
            raise ValueError("field_dim must be a positive integer")
        self.title_dim = schema.get("title_dim", 0)
        self.output_dim = 3 * field_dim + self.title_dim
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
        projected = [self.projections[name](torch.cat([
            values[name], observed[name][:, None].to(values[name].dtype)], dim=-1))
            for name in FIELD_NAMES]
        if self.title_dim:
            projected.append(fields["title"])
        return torch.cat(projected, dim=-1)


class FieldHeads(nn.Module):
    def __init__(self, schema, input_dim):
        super().__init__()
        self.heads = nn.ModuleDict({"brand": nn.Linear(input_dim, len(schema["brand_vocabulary"])),
            "categories": nn.Linear(input_dim, len(schema["category_vocabulary"])),
            "price": nn.Linear(input_dim, 1)})
        self.field_names = tuple(schema["fields"])
        if "title" in self.field_names:
            self.heads["title"] = nn.Linear(input_dim, schema["title_dim"])

    def forward(self, decoded):
        return {name: head(decoded) for name, head in self.heads.items()}

    def losses(self, predictions, fields):
        masks = {"brand": fields["brand"] > 0, "categories": fields["categories_observed"],
                 "price": fields["price_observed"]}
        if "title" in self.field_names:
            masks["title"] = fields["title_observed"]
        losses, counts = {}, {}
        for name in self.field_names:
            mask = masks[name]
            counts[name] = int(mask.sum())
            if not counts[name]:
                losses[name] = predictions[name].sum() * 0
            elif name == "brand":
                losses[name] = F.cross_entropy(predictions[name][mask], fields[name][mask] - 1)
            elif name == "categories":
                losses[name] = F.binary_cross_entropy_with_logits(predictions[name][mask], fields[name][mask])
            elif name == "title":
                losses[name] = F.mse_loss(predictions[name][mask], fields[name][mask].detach())
            else:
                losses[name] = F.huber_loss(predictions[name][mask], fields[name][mask], delta=1.0)
        return losses, counts
