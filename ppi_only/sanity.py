"""Mandatory sanity checks — run BEFORE any training (gate)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from . import config as CFG
from .compat import build_criterion, load_domain_csvs
from .datasets import VectorDataset, ImageDataset, build_loader
from .graph import build_relations_dict, make_vector, count_null_vectors
from .image import outer_diff_image
from .models import build_model


def run_sanity(cfg: Dict[str, Any], data_dir: Path, ppi_csv: Path) -> bool:
    print(f"\n=== SANITY CHECKS [{cfg['domain']} | {cfg['model']} | pooling={cfg['pooling']}] ===")
    ok = True
    domain = cfg["domain"]
    df_tr, df_val, ic_df, _ = load_domain_csvs(domain, data_dir)
    terms = list(df_tr.columns[2:])
    n_classes = len(terms)
    ids_tr = df_tr["ID"].astype(str).tolist()
    ids_val = df_val["ID"].astype(str).tolist()
    ic_vec = ic_df.set_index("terms").reindex(terms)["IC"].fillna(0).to_numpy(np.float32)

    rel, id2idx = build_relations_dict(ppi_csv, ids_tr)
    train_size = len(ids_tr)

    # 1) alignment: raw vector length == N; no val id leaks into the train index base
    v_raw = make_vector(ids_tr[0], rel, train_size, "raw", train_size)
    c1 = (len(v_raw) == train_size) and not (set(ids_val) & set(id2idx.keys()))
    print(f"[1] alignment: raw_len={len(v_raw)}==N({train_size}) & no val in train-base -> {c1}")
    ok &= c1

    # 2) isolation (should be ~33-38%)
    n_null_tr = count_null_vectors(ids_tr, rel, train_size)
    n_null_val = count_null_vectors(ids_val, rel, train_size)
    print(f"[2] isolated: train={n_null_tr}/{train_size} ({100*n_null_tr/train_size:.1f}%) "
          f"| val={n_null_val}/{len(ids_val)} ({100*n_null_val/len(ids_val):.1f}%)")

    # 3) shapes / ranges of a batch
    y_tr = df_tr.iloc[:, 2:].to_numpy(np.float32)
    if CFG.is_image_head(cfg["model"]):
        ds = ImageDataset(ids_tr[:64], y_tr[:64], rel, train_size, cfg["pooling"], int(cfg["image_size"]))
    else:
        os_ = train_size if cfg["pooling"] == "raw" else int(cfg["out_size"])
        ds = VectorDataset(ids_tr[:64], y_tr[:64], rel, train_size, cfg["pooling"], os_)
    dl = build_loader(ds, 8, False, 0, cfg["seed"])
    xb, yb = next(iter(dl))
    c3 = (yb.shape[1] == n_classes)
    rng_ok = True
    if CFG.is_image_head(cfg["model"]):
        rng_ok = bool(xb.min() >= -1.01 and xb.max() <= 1.01)  # Normalize(0.5,0.5) -> ~[-1,1]
    print(f"[3] batch X={tuple(xb.shape)} y={tuple(yb.shape)} classes_ok={c3} img_range_ok={rng_ok}")
    ok &= c3 and rng_ok

    # 4) image symmetry + zero diagonal + no NaN
    if CFG.is_image_head(cfg["model"]):
        vv = make_vector(ids_tr[0], rel, train_size, cfg["pooling"], int(cfg["image_size"]))
        M = outer_diff_image(vv).astype(np.float32)
        c4 = np.allclose(M, M.T) and np.all(np.diag(M) == 0) and not np.isnan(M).any()
        print(f"[4] image symmetric & zero-diag & finite -> {c4}")
        ok &= c4
    else:
        print("[4] image check skipped (vector head)")

    # 5) trainable params
    input_dim = train_size if cfg["pooling"] == "raw" else int(cfg["out_size"])
    model = build_model(cfg, input_dim, n_classes)
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[5] params total={tot:,} trainable={tr:,}")

    # 6) no NaN in loss for one forward, for both losses
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    xb = xb.to(device)
    yb = yb.to(device)
    c6 = True
    for loss_name in ("protein", "bce"):
        crit = build_criterion(loss_name, ic_vec, device, cfg["pos_weight_mode"])
        with torch.no_grad():
            logits = model(xb)
            loss = crit(logits, yb)
        finite = bool(torch.isfinite(loss).item())
        print(f"[6] loss={loss_name}: value={float(loss):.4f} finite={finite}")
        c6 &= finite
    ok &= c6

    print(f"=== SANITY {'PASSED' if ok else 'FAILED'} ===\n")
    return bool(ok)
