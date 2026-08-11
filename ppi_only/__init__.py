"""PPI-only (legacy) refactored package.

Reconstructs the Qualifying-Exam PPI-only pipeline as clean, reproducible scripts.
Design constraints (see instrucoes_experimentos.md):
  - Feature = per-protein PPI score profile against the TRAIN set only (anti-leakage).
  - NO ProtT5 channels; NO equalize / dilate / enhance / convexize.
  - Metrics & loss are REUSED from the v32 package (ppi_v4) — never reimplemented.
"""
__all__ = ["config", "graph", "image", "datasets", "models", "sanity", "train"]
