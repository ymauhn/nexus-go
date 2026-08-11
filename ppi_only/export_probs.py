"""Exporta as probabilidades de um run ja treinado, a partir do checkpoint.

O `run_experiment` calcula as probs para avaliar e as descarta — so as metricas sobrevivem. Isso
basta para as tabelas, mas nao para analises por subconjunto de proteinas (grau no grafo, isoladas
vs conectadas), que precisam da matriz completa.

Este script NAO retreina e NAO recalcula metricas por conta propria: reconstroi exatamente o mesmo
dataset e modelo que `run_experiment` montou, carrega `runs/{tag}_best.pt` e chama o mesmo
`predict_proba`. Como checagem, recalcula as metricas e compara com o `{tag}_metrics.json` gravado
no treino — se divergirem, algo na reconstrucao esta errado e o script aborta.

Uso:
    python -m ppi_only.export_probs --domain bp --model mlp --pooling raw
    python -m ppi_only.export_probs --tag bp_mlp_raw_protein_seed1337
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as CFG
from .compat import (evaluate_collect, generate_ontology, load_domain_csvs,
                     predict_proba, set_global_seed)
from .graph import build_relations_dict
from .datasets import VectorDataset, ImageDataset, build_loader
from .models import build_model

TOL = 1e-4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=["bp", "cc", "mf"])
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--pooling", default="raw")
    ap.add_argument("--loss-name", default="protein")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--ppi-csv", type=Path, default=None)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: --results-dir")
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args(argv)

    ppi_csv = args.ppi_csv or (args.data_dir / "ppi.csv")
    out_dir = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CFG.load_config(args.config, domain=args.domain, model=args.model,
                          pooling=args.pooling, loss_name=args.loss_name)
    tag = CFG.make_tag(cfg)
    seed = int(cfg["seed"])
    set_global_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_image = CFG.is_image_head(cfg["model"])

    ck = args.runs_dir / f"{tag}_best.pt"
    if not ck.exists():
        raise SystemExit(f"checkpoint ausente: {ck}")

    # Reconstrucao identica a de run_experiment. Qualquer divergencia aqui aparece na checagem
    # de metricas ao final.
    df_tr, df_val, ic_df, go_path = load_domain_csvs(args.domain, args.data_dir)
    df_test = pd.read_csv(args.data_dir / f"{args.domain}_test.csv")
    terms = list(df_tr.columns[2:])
    n_classes = len(terms)
    ic_dict = ic_df.set_index("terms")["IC"].to_dict()
    ontology = generate_ontology(go_path, specific_space=True,
                                 name_specific_space=CFG.DOM_INFO[args.domain]["type"])
    root = CFG.DOM_INFO[args.domain]["root"]

    ids_tr = df_tr["ID"].astype(str).tolist()
    rel, _ = build_relations_dict(ppi_csv, ids_tr)
    train_size = len(ids_tr)
    input_dim = None if is_image else (train_size if cfg["pooling"] == "raw" else int(cfg["out_size"]))

    model = build_model(cfg, input_dim or 0, n_classes).to(device)
    model.load_state_dict(torch.load(ck, map_location=device))
    print(f"[{tag}] checkpoint carregado, device={device.type}, input_dim={input_dim}")

    gravado = {}
    mp = args.results_dir / f"{tag}_metrics.json"
    if mp.exists():
        gravado = json.loads(mp.read_text(encoding="utf-8"))

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        df = {"val": df_val, "test": df_test, "train": df_tr}[split]
        ids = df["ID"].astype(str).tolist()
        y = df.iloc[:, 2:].to_numpy(dtype=np.float32)
        if is_image:
            ds = ImageDataset(ids, y, rel, train_size, cfg["pooling"], int(cfg["image_size"]),
                              cache=bool(int(cfg.get("vec_cache", 1))))
        else:
            ds = VectorDataset(ids, y, rel, train_size, cfg["pooling"], input_dim,
                               cache=bool(int(cfg.get("vec_cache", 1))))
        dl = build_loader(ds, cfg["batch_size"], False, cfg["num_workers"], seed)

        probs, gts = predict_proba(dl, model, device)
        np.save(out_dir / f"probs_{tag}_{split}.npy", probs.astype(np.float32))
        np.save(out_dir / f"gtppi_{args.domain}_{split}.npy", gts.astype(np.int8))

        m = evaluate_collect(probs, gts, ont_names=terms, ontology=ontology,
                             ic=ic_dict, root=root)
        ref = gravado.get(split)
        if ref:
            piores = {k: (float(m[k]), float(ref[k])) for k in ("fmax", "wfmax", "smin")
                      if abs(float(m[k]) - float(ref[k])) > TOL}
            if piores:
                raise SystemExit(
                    f"[{tag}/{split}] metricas recalculadas divergem das gravadas no treino: "
                    f"{piores}. A reconstrucao do dataset ou do modelo nao confere — nao use "
                    f"estas probs.")
            print(f"  [{split}] {probs.shape}  confere com o treino "
                  f"(wfmax={m['wfmax']:.4f}, fmax={m['fmax']:.4f})")
        else:
            print(f"  [{split}] {probs.shape}  wfmax={m['wfmax']:.4f} "
                  f"(sem metrics.json para conferir)")

    print(f"[OK] probs em {out_dir}/probs_{tag}_*.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
