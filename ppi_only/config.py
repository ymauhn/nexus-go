"""Central defaults + YAML/CLI config merging for the PPI-only pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

# Gene Ontology domain metadata (roots used to compute Fmax*).
DOM_INFO = {
    "bp": {"type": "biological_process", "root": "GO:0008150"},
    "cc": {"type": "cellular_component", "root": "GO:0005575"},
    "mf": {"type": "molecular_function", "root": "GO:0003674"},
}

DEFAULTS: Dict[str, Any] = dict(
    # data / task
    domain="bp",
    # model: "mlp" | "cnn1d" | "resnet50" | "convnext_tiny"
    model="mlp",
    # pooling of the train-score vector: "max" | "avg" | "raw"
    #   raw = NO pooling -> classifier consumes the full N-dim vector (N = train_size).
    #   (raw is only valid for vector heads: mlp / cnn1d)
    pooling="max",
    loss_name="protein",        # "protein" | "bce"
    pos_weight_mode="ic",       # "none" | "ic" (BCE pos_weight = IC)
    out_size=1024,              # vector length after max/avg pooling
    image_size=224,             # image head: vector pooled to this -> outer-diff (image_size x image_size)
    # training
    batch_size=32,
    lr=1e-4,
    max_epochs=200,
    patience=25,
    num_workers=0,              # 0 keeps ordering fully deterministic; raise if I/O bound
    # vec_cache=1 precomputa os vetores no __init__ do dataset em vez de recomputar a cada época.
    # make_vector é função PURA (sem RNG), então é bit a bit transparente. Medido nesta base:
    # 283 us/amostra em cc = ~1,2 h só de preparo de dados em 200 épocas. vec_cache=0 desliga
    # (usado para provar a equivalência). Não se aplica a pooling='raw', que já é barato.
    vec_cache=1,
    # image backbone
    unfreeze="all",             # "none" | "last2" | "all"
    pretrained=1,
    # reproducibility
    seed=1337,
    # smoke / debugging
    smoke=False,
    limit_train=0,              # 0 = use all; >0 = subset for smoke tests
    limit_val=0,
)

_VECTOR_HEADS = {"mlp", "cnn1d"}
_IMAGE_HEADS = {"resnet50", "convnext_tiny", "efficientnet_b0", "vit_b_16"}


def _coerce(v: str) -> Any:
    """Parse a CLI --override value (key=value) into int/float/bool/str."""
    low = v.lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def load_config(config_path: Path | None,
                domain: str | None = None,
                model: str | None = None,
                overrides: List[str] | None = None,
                **cli_kwargs: Any) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if config_path is not None:
        with open(config_path, "r") as f:
            cfg.update(yaml.safe_load(f) or {})
    if domain is not None:
        cfg["domain"] = domain
    if model is not None:
        cfg["model"] = model
    for k, v in cli_kwargs.items():
        if v is not None:
            cfg[k] = v
    for ov in (overrides or []):
        if "=" not in ov:
            raise ValueError(f"--override precisa ser key=value, recebi: {ov!r}")
        k, v = ov.split("=", 1)
        cfg[k.strip()] = _coerce(v.strip())
    _validate(cfg)
    return cfg


def _validate(cfg: Dict[str, Any]) -> None:
    if cfg["domain"] not in DOM_INFO:
        raise ValueError(f"domain inválido: {cfg['domain']}")
    if cfg["model"] not in (_VECTOR_HEADS | _IMAGE_HEADS):
        raise ValueError(f"model inválido: {cfg['model']}")
    if cfg["pooling"] not in ("max", "avg", "raw"):
        raise ValueError(f"pooling inválido: {cfg['pooling']}")
    if cfg["pooling"] == "raw" and cfg["model"] in _IMAGE_HEADS:
        raise ValueError("pooling='raw' só é válido para heads vetoriais (mlp/cnn1d), não para imagem.")
    if cfg["loss_name"] not in ("protein", "bce"):
        raise ValueError(f"loss_name inválido: {cfg['loss_name']}")


def is_image_head(model: str) -> bool:
    return model in _IMAGE_HEADS


def make_tag(cfg: Dict[str, Any]) -> str:
    """Run tag: {domain}_{model}_{pooling}_{loss}_seed{seed}[_smoke].

    O sufixo `_smoke` é OBRIGATÓRIO quando cfg['smoke'] está ligado. Sem ele, um gate de mockup
    (1 época, 512 amostras) grava em results/{tag}_metrics.json e runs/{tag}_best.pt com
    exatamente o mesmo nome do run real — destruindo em silêncio um resultado de 200 épocas.
    Aconteceu de verdade: o mock de bp_cnn1d_max sobrescreveu wFmax(val) 0,3289 por 0,2081.
    """
    tag = f"{cfg['domain']}_{cfg['model']}_{cfg['pooling']}_{cfg['loss_name']}_seed{cfg['seed']}"
    if cfg.get("_rawresize"):
        tag += "_rawresize"
    if cfg.get("smoke"):
        tag += "_smoke"
    return tag
