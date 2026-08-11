"""PPI-only feature construction: train-restricted score profile + pooling.

Faithful to the legacy notebook, WITHOUT the optional histogram equalization.
Anti-leakage safeguard: the feature basis is the TRAIN set only. A protein is a
"null row" (isolated) iff it has no TRAIN neighbor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def build_relations_dict(ppi_csv: Path, training_ids) -> Tuple[Dict, Dict]:
    """Build the train-restricted adjacency (relation) dict.

    - id2idx indexes TRAIN proteins only -> vector dimension N = len(training_ids).
    - undirected dedup keeping MAX score; raw self-loops dropped.
    - a neighbor is registered only if it belongs to the TRAIN set.
    - a self-loop of weight 1.0 is added ONLY for train proteins (diagonal entry);
      this does NOT add a node/dimension (adjacency stays N x N, not (N+1)x(N+1)).
    """
    df = pd.read_csv(ppi_csv)
    u = df.iloc[:, 0].astype(str).values
    v = df.iloc[:, 1].astype(str).values
    s = pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0.0).values.astype(float)

    training_ids = [str(t) for t in training_ids]
    set_train = set(training_ids)
    id2idx = {pid: i for i, pid in enumerate(training_ids)}

    # dedup undirected pair {a,b} keeping the max score, ignoring self-loops
    edge_scores: Dict[Tuple[str, str], float] = {}
    for a, b, w in zip(u, v, s):
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        w = float(w)
        prev = edge_scores.get(key)
        if prev is None or w > prev:
            edge_scores[key] = w

    rel: Dict[str, Dict[str, list]] = {}

    def ensure(pid: str) -> None:
        if pid not in rel:
            rel[pid] = {"relation": []}

    for (a, b), w in edge_scores.items():
        ensure(a)
        ensure(b)
        if b in set_train:
            rel[a]["relation"].append([id2idx[b], w])
        if a in set_train:
            rel[b]["relation"].append([id2idx[a], w])

    # self-loop 1.0 for train proteins (diagonal); dimension stays N
    for pid in training_ids:
        ensure(pid)
        rel[pid]["relation"].append([id2idx[pid], 1.0])

    return rel, id2idx


def pool_to_size(feat: np.ndarray, out_size: int, mode: str) -> np.ndarray:
    """Pool the N-dim train-score vector.

    mode='max'/'avg' -> length out_size (1D pooling).
    mode='raw'       -> NO pooling: returns the full N-dim vector unchanged.
    """
    if mode == "raw":
        return feat.astype(np.float32, copy=False)
    v = torch.tensor(feat, dtype=torch.float32)[None, None, :]
    k = max(1, feat.shape[0] // out_size)
    p = F.max_pool1d(v, k) if mode == "max" else F.avg_pool1d(v, k)
    out = p.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    if out.shape[0] != out_size:
        out = np.resize(out, out_size).astype(np.float32)
    return out


def make_vector(pid: str, rel: Dict, train_size: int, pooling: str, out_size: int) -> np.ndarray:
    """Score profile of `pid` against the TRAIN set, then pooled. No equalization."""
    feat = np.zeros(train_size, dtype=np.float32)
    node = rel.get(pid)
    if node is not None:
        for j, w in node["relation"]:
            if 0 <= j < train_size:
                if w > feat[j]:
                    feat[j] = float(w)
    return pool_to_size(feat, out_size, pooling)


def count_null_vectors(ids: List[str], rel: Dict, train_size: int) -> int:
    """Number of proteins whose train-score profile is all-zeros (isolated)."""
    n = 0
    for pid in ids:
        node = rel.get(pid)
        has = False
        if node is not None:
            for j, w in node["relation"]:
                if 0 <= j < train_size and w > 0:
                    has = True
                    break
        if not has:
            n += 1
    return n
