import argparse
import yaml
import torch
import json
from pathlib import Path

from ppi_v4.config import DEFAULT_CFG, RUNS_DIR, DOM_INFO
from ppi_v4.engine import build_criterion, fit, predict_proba, set_global_seed
from ppi_v4.data import load_domain_csvs
from ppi_v4.graph_utils import load_channel, build_sparse_graph
from ppi_v4.models_v32 import UnifiedProteinModel
from ppi_v4.metrics import evaluate_collect, generate_ontology

def _build_index_loaders(cfg: dict, data_dir: Path, all_ids: list[str], eval_split: str):
    import numpy as np
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader

    class _IndexDS(Dataset):
        def __init__(self, ids, labels, name2idx):
            self.ids = list(ids)
            self.y = labels.astype(np.float32)
            self.idx = np.array([name2idx[i] for i in self.ids], dtype=np.int64)
        def __len__(self): return len(self.ids)
        def __getitem__(self, i): return (torch.tensor(self.idx[i], dtype=torch.long), torch.from_numpy(self.y[i]))

    domain = cfg["domain"]
    df_tr, df_val_default, ic_df, go_path = load_domain_csvs(domain, data_dir)
    df_eval = pd.read_csv(Path(data_dir) / f"{domain}_test.csv") if eval_split == "test" else df_val_default
    terms = df_tr.columns[2:]
    y_tr, y_ev = df_tr.iloc[:, 2:].to_numpy(np.float32), df_eval.iloc[:, 2:].to_numpy(np.float32)
    train_ids, eval_ids = df_tr["ID"].astype(str).values, df_eval["ID"].astype(str).values
    
    ic_vec = ic_df.set_index("terms").reindex(terms)["IC"].fillna(0).to_numpy(dtype=np.float32)
    ic_dict = ic_df.set_index("terms")["IC"].to_dict()
    ont = generate_ontology(go_path, specific_space=True, name_specific_space=DOM_INFO[domain]["type"])

    name2idx = {pid: i for i, pid in enumerate(all_ids)}
    ds_tr, ds_ev = _IndexDS(train_ids, y_tr, name2idx), _IndexDS(eval_ids, y_ev, name2idx)
    dl_kwargs = dict(batch_size=cfg["batch_size"], num_workers=cfg.get("num_workers", 4), pin_memory=True)
    
    return DataLoader(ds_tr, **{**dl_kwargs, "shuffle": True}), DataLoader(ds_ev, **{**dl_kwargs, "shuffle": False}), len(terms), terms, ic_vec, ic_dict, ont

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default="data")
    ap.add_argument("--ppi-csv", type=Path, default="data/ppi.csv")
    args = ap.parse_args()

    with open(args.config, "r") as f: cfg = yaml.safe_load(f)
    cfg_full = DEFAULT_CFG.copy()
    cfg_full.update(cfg)
    
    set_global_seed(cfg_full["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- NOMES E CAMINHOS 100% DINÂMICOS ---
    domain = cfg_full.get("domain", "bp")
    current_raw_dir = Path(f"raw/raw_{domain}") # Ajusta pasta raw de acordo com o domínio
    method = cfg_full.get("method", "gnn")
    clf = cfg_full.get("classifier_type", "mlp")
    gnn_type = cfg_full.get("gnn_type", "gat")
    thr = cfg_full.get("thr", 0.40)
    
    # Formata o limiar para não ter ponto no arquivo (ex: 0.40 vira 040)
    thr_str = f"{thr:.2f}".replace(".", "")
    
    if method in ["gnn", "load_gnn"]:
        run_tag = f"{domain}_{method}_{gnn_type}_thr{thr_str}_{clf}_v32_{cfg_full['split']}"
    else:
        run_tag = f"{domain}_{method}_thr{thr_str}_{clf}_v32_{cfg_full['split']}"
    # --------------------------------------

    import pandas as pd
    df_tr, df_val, _, _ = load_domain_csvs(domain, args.data_dir)
    df_ev = pd.read_csv(args.data_dir / f"{domain}_test.csv") if cfg_full["split"] == "test" else df_val
    all_ids = list(dict.fromkeys(df_tr["ID"].astype(str).tolist() + df_ev["ID"].astype(str).tolist()))
    name2idx = {pid: i for i, pid in enumerate(all_ids)}

    # Carrega Embeddings
    H0 = torch.cat([load_channel(current_raw_dir, all_ids, ch, device) for ch in ["24", "23", "22"]], dim=1)
    
    A_sparse, edge_index, edge_weight = None, None, None
    if method in ["fixed", "hybrid", "gnn"]:
        A_sparse, edge_index, edge_weight = build_sparse_graph(args.ppi_csv, name2idx, cfg_full["thr"], cfg_full["norm"], bool(cfg_full["convexize"]), device)

    # Configura Loaders e Instancia Modelo
    dl_tr, dl_ev, ncls, terms, ic_vec, ic_dict, ont = _build_index_loaders(cfg_full, args.data_dir, all_ids, cfg_full["split"])
    model = UnifiedProteinModel(cfg_full, H0, A_sparse, edge_index, edge_weight, n_classes=ncls).to(device)

    # Inicia Treinamento
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg_full["lr"])
    crit = build_criterion(cfg_full["loss_name"], ic_vec, device, cfg_full["pos_weight_mode"])
    
    best_path = fit(model, dl_tr, dl_ev, crit, opt, device, cfg_full["max_epochs"], cfg_full["patience"], run_tag, RUNS_DIR)

    if best_path and Path(best_path).exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        
    # Salva os embeddings com as tags exatas do experimento!
    if method == "gnn":
        gnn_emb = model.extract_gnn_features()
        # Ex: precomp/gnn_gat_bp_thr040_vision_hidden.pt
        out_path = Path("precomp") / f"gnn_{gnn_type}_{domain}_thr{thr_str}_{clf}_hidden.pt"
        out_path.parent.mkdir(exist_ok=True)
        torch.save(gnn_emb, out_path)
        print(f"✅ Embeddings ({gnn_type.upper()}, E2E com {clf.upper()}) salvos em {out_path}!")

    print("\n--- AVALIAÇÃO FINAL ---")
    probs, gts = predict_proba(dl_ev, model, device)
    metrics = evaluate_collect(probs, gts, ont_names=list(terms), ontology=ont, ic=ic_dict, root=DOM_INFO[domain]["root"])
    
    print(f"Métricas ({cfg_full['split']}):", metrics)
    (RUNS_DIR / f"{run_tag}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()