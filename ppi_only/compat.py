"""Bridge to the v32 package (ppi_v4) — REUSE metrics/losses/engine/backbones.

We do NOT reimplement the metric or the loss: this module locates the existing
`ppi_v4` package (expected at the project root) and re-exports exactly the
functions the PPI-only pipeline needs. If the package is not importable, we try
to add plausible project roots to sys.path before failing with a clear message.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_ppi_v4_importable() -> None:
    try:
        import ppi_v4.ppi_v4  # noqa: F401
        return
    except Exception:
        pass
    here = Path(__file__).resolve()
    candidates = [here.parents[1], here.parents[2] if len(here.parents) > 2 else here.parents[1], Path.cwd()]
    for root in candidates:
        if (root / "ppi_v4" / "ppi_v4" / "engine.py").exists():
            sys.path.insert(0, str(root))
            return
    raise ImportError(
        "Não encontrei o pacote `ppi_v4` (esperado em <raiz_do_projeto>/ppi_v4/ppi_v4/). "
        "Coloque o pacote v32 na raiz do projeto — ele fornece metrics.py, losses.py, "
        "engine.py e models.py, que esta pipeline REUTILIZA (não reimplementa)."
    )


_ensure_ppi_v4_importable()

# Forward-compat: np.trapz was removed in NumPy 2.0 (renamed np.trapezoid). The reused
# v32 metrics.py calls np.trapz; alias it if the installed NumPy is 2.x. Harmless on 1.x.
import numpy as _np  # noqa: E402
if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid  # type: ignore[attr-defined]

from ppi_v4.ppi_v4.engine import (  # noqa: E402
    set_global_seed,
    build_criterion,
    fit,
    predict_proba,
)
from ppi_v4.ppi_v4.metrics import evaluate_collect, generate_ontology  # noqa: E402
from ppi_v4.ppi_v4.models import build_image_backbone  # noqa: E402
from ppi_v4.ppi_v4.data import load_domain_csvs  # noqa: E402

__all__ = [
    "set_global_seed", "build_criterion", "fit", "predict_proba",
    "evaluate_collect", "generate_ontology", "build_image_backbone",
    "load_domain_csvs",
]
