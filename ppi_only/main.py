"""CLI entry point for a single PPI-only experiment.

Examples:
  python -m ppi_only.main --config configs/ppi_only/mock_mlp.yaml --domain bp --smoke
  python -m ppi_only.main --domain cc --model cnn1d --pooling avg
  python -m ppi_only.main --domain mf --model convnext_tiny --pooling max --override lr=5e-5
  python -m ppi_only.main --sanity-only --domain bp --model resnet50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config as CFG
from .sanity import run_sanity
from .train import run_experiment


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PPI-only single-experiment runner")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--domain", choices=["bp", "cc", "mf"], default=None)
    ap.add_argument("--model", default=None,
                    help="mlp | cnn1d | resnet50 | convnext_tiny")
    ap.add_argument("--arch", dest="model", default=None, help="alias de --model")
    ap.add_argument("--pooling", choices=["max", "avg", "raw"], default=None)
    ap.add_argument("--loss", dest="loss_name", choices=["protein", "bce"], default=None)
    ap.add_argument("--smoke", action="store_true", help="teste rápido: 1 época + subset")
    ap.add_argument("--sanity-only", action="store_true", help="roda só os sanity checks e sai")
    ap.add_argument("--limit-train", type=int, default=None, dest="limit_train")
    ap.add_argument("--limit-val", type=int, default=None, dest="limit_val")
    ap.add_argument("--override", action="append", default=[],
                    help="key=value (repetível) para sobrescrever qualquer campo do cfg")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--ppi-csv", type=Path, default=Path("data/ppi.csv"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    cli_kwargs = dict(loss_name=args.loss_name, pooling=args.pooling,
                      limit_train=args.limit_train, limit_val=args.limit_val)
    if args.smoke:
        cli_kwargs.setdefault("smoke", True)
        cli_kwargs["smoke"] = True
        # smoke defaults (only if not explicitly overridden)
        if args.limit_train is None:
            cli_kwargs["limit_train"] = 512
        if args.limit_val is None:
            cli_kwargs["limit_val"] = 256
        cli_kwargs["max_epochs"] = 1
        cli_kwargs["patience"] = 1

    cfg = CFG.load_config(args.config, domain=args.domain, model=args.model,
                          overrides=args.override, **cli_kwargs)

    print(f">>> cfg: {CFG.make_tag(cfg)} | image_head={CFG.is_image_head(cfg['model'])}")

    # sanity gate
    passed = run_sanity(cfg, args.data_dir, args.ppi_csv)
    if not passed:
        print("!! SANITY FALHOU — abortando antes de treinar. Corrija e rode de novo.")
        return 2
    if args.sanity_only:
        return 0

    result = run_experiment(cfg, args.data_dir, args.ppi_csv, args.runs_dir, args.results_dir)
    print("\n--- RESULT ---")
    for split, m in result["metrics_by_split"].items():
        print(f"  [{split}] wfmax={m['wfmax']:.4f} fmax={m['fmax']:.4f} "
              f"fmax*={m['fmax_star']:.4f} smin={m['smin']:.4f} "
              f"auprc={m['auprc']:.4f} iauprc={m['iauprc']:.4f}")
    print(f"tag={result['tag']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
