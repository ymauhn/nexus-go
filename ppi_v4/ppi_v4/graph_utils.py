import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

def load_channel(raw_dir: Path, ids: List[str], ch: str, device: torch.device, dim: int = 1024) -> torch.Tensor:
    raw_dir = Path(raw_dir)
    X = []
    for pid in ids:
        p = raw_dir / f"{pid}-{ch}.npy"
        if p.exists():
            X.append(np.load(p).astype(np.float32, copy=False))
        else:
            X.append(np.zeros(dim, dtype=np.float32))
    return torch.tensor(np.stack(X, axis=0), dtype=torch.float32, device=device)

def coalesce(idx: torch.Tensor, val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if idx.numel() == 0: return idx, val
    i = idx[0].cpu().numpy().astype(np.int64, copy=False)
    j = idx[1].cpu().numpy().astype(np.int64, copy=False)
    v = val.cpu().numpy().astype(np.float32, copy=False)
    N = int(max(i.max(), j.max()) + 1)
    key = i * (N + 1) + j
    order = np.argsort(key, kind="mergesort")
    _, start = np.unique(key[order], return_index=True)
    sums = np.add.reduceat(v[order], start)
    return torch.tensor(np.vstack([i[order][start], j[order][start]]), dtype=torch.long), torch.tensor(sums, dtype=torch.float32)

def build_sparse_graph(ppi_csv: Path, name2idx: dict, thr: float, norm: str, convexize: bool, device: torch.device):
    df = pd.read_csv(ppi_csv)
    a, b = df.iloc[:, 0].astype(str).to_numpy(), df.iloc[:, 1].astype(str).to_numpy()
    s = pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    src, dst, val = [], [], []
    for u, v, w in zip(a, b, s):
        if u == v or w < thr: continue  # Proteção anti self-loop e threshold
        iu, iv = name2idx.get(u), name2idx.get(v)
        if iu is not None and iv is not None:
            src.extend([iu, iv])
            dst.extend([iv, iu])
            val.extend([float(w), float(w)])

    if not src:
        idx = torch.empty((2, 0), dtype=torch.long)
        val = torch.empty((0,), dtype=torch.float32)
    else:
        idx, val = coalesce(torch.tensor([src, dst], dtype=torch.long), torch.tensor(val, dtype=torch.float32))

    if convexize:
        idx_sym = torch.cat([idx, torch.stack([idx[1], idx[0]], dim=0)], dim=1)
        val_sym = torch.cat([val, val], dim=0)
        idx, val = coalesce(idx_sym, val_sym)

    N = len(name2idx)
    i, j = idx[0].to(device), idx[1].to(device)
    val = val.to(device)
    
    denom_idx = i if norm == "row" else j
    denom = torch.zeros(N, dtype=val.dtype, device=device).index_add(0, denom_idx, val).clamp_min_(1e-12)
    val_n = val / denom[denom_idx]
    
    A = torch.sparse_coo_tensor(torch.stack([i, j], dim=0), val_n, (N, N)).coalesce()
    return A, idx.to(device), val.to(device)