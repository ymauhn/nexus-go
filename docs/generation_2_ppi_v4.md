# Second generation — `ppi_v4/`

The proposed method. Where the first generation represented a protein by its row in the
interaction matrix, this one starts from the sequence and uses the network only to
*contextualise* it.

## Representation

Frozen ProtT5-XL-U50 embeddings, 3,072-d per protein (see [`data.md`](data.md)). The
encoder is never fine-tuned; every experiment reads the same cached vectors, so differences
between runs are attributable to propagation and head alone.

## Propagation — the design axis

The graph is column-normalised, `P = W · Dc⁻¹` with a floor of ε = 10⁻¹², and the fixed
operator is

```
H⁽ʰ⁾ = α · H⁽ʰ⁻¹⁾ + (1 − α) · Pᵀ · H⁽ʰ⁻¹⁾ ,    h = 1 … T
```

with no trainable parameters. It is computed once per experiment, outside the training
loop, through sparse–dense multiplication in COO format, and cached. `T = 0` (equivalently
α = 1) is the internal sequence-only control.

The learned alternative is a forward pass over the full graph with `GCNConv`, `GATConv`
(four attention heads in the hidden layers, edge attribute = unnormalised combined score)
or `SAGEConv` from PyTorch Geometric. `models_v32.py` imports them inside a
`try/except ImportError`, so the package remains usable without PyTorch Geometric when
`method` is not `gnn`.

## Heads

- **Vector** — linear → batch norm → ReLU → dropout, with a final linear layer emitting one
  logit per GO term.
- **Visual** — the vector is clipped to [−50, 50], passed through the logistic function and
  split into three blocks of 1,024 reshaped to 32 × 32 channels, one per ProtT5 layer, then
  consumed by ResNet-50 or ConvNeXt-Tiny. No image is ever written to disk.

## Objective

`ProteinLoss`, a composite inherited from the InterLabelGO+ line:

```
L = L_ZLPR · (1 − wF1_GO) · (1 − wF1_protein)
```

weighted by information content, with `BCEWithLogitsLoss(pos_weight=IC)` available as the
ablation comparison.

## Fusion

Early fusion concatenates (4,096-d) or adds (3,072-d, summing the convolutional
representation into each of the three layer blocks). Late fusion operates on saved
probabilities: linear combination with γ, pointwise minimum and maximum, per-term weighting
by IC, per-protein weighting by node degree, a combined weighting with λ, and stacking with
a decision tree, logistic regression or MLP.

## Configuration

Everything is driven from YAML in `configs/ppi_v4/`:

| File | What it runs |
|---|---|
| `01_fixo_vision.yaml` | fixed propagation, visual head |
| `02_fixo_mlp.yaml` | fixed propagation, MLP head |
| `03_gnn_puro.yaml`, `03_gnn_vision_e2e.yaml` | end-to-end graph learning |
| `04_hibrido_mlp.yaml`, `04_load_gnn_vision.yaml` | hybrid, reusing precomputed GNN embeddings |
| `05_hybrid_add_vision.yaml`, `06_hybrid_concat_mlp.yaml` | early-fusion variants |
