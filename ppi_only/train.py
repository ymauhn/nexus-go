"""Train a single PPI-only experiment and save all artifacts.

Reuses v32 engine.fit / predict_proba / build_criterion and metrics.evaluate_collect.
Artifacts per run (tag = {domain}_{model}_{pooling}_{loss}_seed{seed}):
  runs/{tag}_best.pt            weights (best val)
  runs/{tag}_losses.png         train-vs-val loss curve   (saved by engine.fit)
  runs/{tag}_loss_hist.json     raw loss history          (saved by engine.fit)
  results/{tag}_metrics.json    6 metrics for val & test
  results/{tag}_metrics.csv     same, tabular
  results/{tag}_manifest.json   full config + env + shapes
"""
from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from . import config as CFG
from .compat import (
    set_global_seed, build_criterion, fit, predict_proba,
    evaluate_collect, generate_ontology, load_domain_csvs,
)
from .datasets import VectorDataset, ImageDataset, build_loader
from .graph import build_relations_dict, count_null_vectors
from .models import build_model

_METRIC_KEYS = ["fmax", "fmax_star", "wfmax", "smin", "auprc", "iauprc"]


def _lib_versions() -> Dict[str, str]:
    v = {"python": platform.python_version(), "torch": torch.__version__}
    try:
        import torchvision
        v["torchvision"] = torchvision.__version__
    except Exception:
        pass
    v["numpy"] = np.__version__
    v["pandas"] = pd.__version__
    return v


def _limit(ids, y, n):
    if n and n > 0:
        return ids[:n], y[:n]
    return ids, y


def run_experiment(cfg: Dict[str, Any], data_dir: Path, ppi_csv: Path,
                   runs_dir: Path, results_dir: Path) -> Dict[str, Any]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["seed"])
    set_global_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = cfg["domain"]
    tag = CFG.make_tag(cfg)
    is_image = CFG.is_image_head(cfg["model"])

    # ---- data ----
    df_tr, df_val, ic_df, go_path = load_domain_csvs(domain, data_dir)
    test_path = Path(data_dir) / f"{domain}_test.csv"
    df_test = pd.read_csv(test_path) if test_path.exists() else None

    terms = list(df_tr.columns[2:])
    n_classes = len(terms)
    ic_vec = ic_df.set_index("terms").reindex(terms)["IC"].fillna(0).to_numpy(dtype=np.float32)
    ic_dict = ic_df.set_index("terms")["IC"].to_dict()
    ontology = generate_ontology(go_path, specific_space=True,
                                 name_specific_space=CFG.DOM_INFO[domain]["type"])
    root = CFG.DOM_INFO[domain]["root"]

    ids_tr = df_tr["ID"].astype(str).tolist()
    y_tr = df_tr.iloc[:, 2:].to_numpy(dtype=np.float32)
    ids_val = df_val["ID"].astype(str).tolist()
    y_val = df_val.iloc[:, 2:].to_numpy(dtype=np.float32)
    if df_test is not None:
        ids_test = df_test["ID"].astype(str).tolist()
        y_test = df_test.iloc[:, 2:].to_numpy(dtype=np.float32)

    if cfg.get("smoke"):
        ids_tr, y_tr = _limit(ids_tr, y_tr, cfg.get("limit_train", 512) or 512)
        ids_val, y_val = _limit(ids_val, y_val, cfg.get("limit_val", 256) or 256)
        if df_test is not None:
            ids_test, y_test = _limit(ids_test, y_test, cfg.get("limit_val", 256) or 256)

    # relation dict is built on the (possibly subset) TRAIN ids
    rel, _ = build_relations_dict(ppi_csv, ids_tr)
    train_size = len(ids_tr)

    # ---- datasets / loaders ----
    # vec_cache: pré-computa os vetores uma vez em vez de a cada época. Numericamente transparente
    # (make_vector é função pura, sem RNG); desligue com vec_cache=0 para conferir a equivalência.
    vec_cache = bool(int(cfg.get("vec_cache", 1)))
    if is_image:
        img = int(cfg["image_size"])
        ds_tr = ImageDataset(ids_tr, y_tr, rel, train_size, cfg["pooling"], img, cache=vec_cache)
        ds_val = ImageDataset(ids_val, y_val, rel, train_size, cfg["pooling"], img, cache=vec_cache)
        ds_test = ImageDataset(ids_test, y_test, rel, train_size, cfg["pooling"], img, cache=vec_cache) if df_test is not None else None
        input_dim = None
    else:
        input_dim = train_size if cfg["pooling"] == "raw" else int(cfg["out_size"])
        ds_tr = VectorDataset(ids_tr, y_tr, rel, train_size, cfg["pooling"], input_dim, cache=vec_cache)
        ds_val = VectorDataset(ids_val, y_val, rel, train_size, cfg["pooling"], input_dim, cache=vec_cache)
        ds_test = VectorDataset(ids_test, y_test, rel, train_size, cfg["pooling"], input_dim, cache=vec_cache) if df_test is not None else None

    dl_tr = build_loader(ds_tr, cfg["batch_size"], True, cfg["num_workers"], seed)
    dl_val = build_loader(ds_val, cfg["batch_size"], False, cfg["num_workers"], seed)
    dl_test = build_loader(ds_test, cfg["batch_size"], False, cfg["num_workers"], seed) if ds_test is not None else None

    # ---- model / optim / loss ----
    model = build_model(cfg, input_dim or 0, n_classes).to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg["lr"])
    crit = build_criterion(cfg["loss_name"], ic_vec, device, cfg["pos_weight_mode"])

    # ---- train (engine.fit saves best.pt + loss curve + hist) ----
    t0 = time.time()
    best_path = fit(model, dl_tr, dl_val, crit, opt, device,
                    int(cfg["max_epochs"]), int(cfg["patience"]), tag, runs_dir, seed=seed)
    train_secs = round(time.time() - t0, 1)
    if best_path and Path(best_path).exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    # ---- evaluate val & test ----
    metrics_by_split: Dict[str, Dict[str, float]] = {}
    eval_loaders = [("val", dl_val)]
    if dl_test is not None:
        eval_loaders.append(("test", dl_test))
    for split_name, loader in eval_loaders:
        probs, gts = predict_proba(loader, model, device)
        m = evaluate_collect(probs, gts, ont_names=terms, ontology=ontology,
                             ic=ic_dict, root=root)
        metrics_by_split[split_name] = {k: float(m[k]) for k in _METRIC_KEYS}

    # ---- persist metrics (json + csv) ----
    (results_dir / f"{tag}_metrics.json").write_text(
        json.dumps(metrics_by_split, indent=2), encoding="utf-8")
    with open(results_dir / f"{tag}_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split"] + _METRIC_KEYS)
        for sp, m in metrics_by_split.items():
            w.writerow([sp] + [m[k] for k in _METRIC_KEYS])

    # ---- manifest ----
    n_null_tr = count_null_vectors(ids_tr, rel, train_size)
    manifest = {
        "tag": tag, "cfg": cfg, "seed": seed, "device": device.type,
        "n_classes": n_classes, "train_size": train_size,
        "input_dim": input_dim, "image_size": cfg["image_size"] if is_image else None,
        "null_train_vectors": n_null_tr,
        "train_seconds": train_secs, "libs": _lib_versions(),
        "metrics_by_split": metrics_by_split,
    }
    (results_dir / f"{tag}_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return {"tag": tag, "domain": domain, "model": cfg["model"],
            "pooling": cfg["pooling"], "loss": cfg["loss_name"],
            "metrics_by_split": metrics_by_split}
