# Capítulo 6 — tabelas geradas

Gerado por `scripts_v4/tabelas_cap6.py` a partir de `experimentos_finais_dissertacao\pasta_mais_importante\tabelas\results_dedupsum.csv` (6 runs, `dedup=sum`).
**Não editar à mão** — regenerar após novas runs. Célula `—` significa run ausente do CSV, nunca valor omitido.

Smin em bits, **menor é melhor**; nas demais métricas, maior é melhor.

### Tabela 6.8 — Ablação de hops, SPMM fixo + MLP (α=0,5) — VALIDAÇÃO

A curva completa de propagação. É a tabela que localiza o melhor número de saltos em cada ontologia, e o argumento de que o ganho satura cedo depende dela.

| hops | BP wFmax | BP Smin ↓ | CC wFmax | CC Smin ↓ | MF wFmax | MF Smin ↓ |
|---|---|---|---|---|---|---|
| 0 | — | — | — | — | — | — |
| 1 | 0,5549 | 17,91 | — | — | — | — |
| 2 | — | — | — | — | — | — |
| 3 | — | — | — | — | — | — |
| 4 | — | — | — | — | — | — |
| 5 | — | — | — | — | — | — |
| 6 | — | — | — | — | — | — |
| 7 | — | — | — | — | — | — |
| 8 | — | — | — | — | — | — |
| 9 | — | — | — | — | — | — |
| 10 | — | — | — | — | — | — |

### Tabela 6.10 — Ablação de α no melhor hop — VALIDAÇÃO

α controla quanto da representação própria do nó sobrevive a cada salto. Para um nó isolado a agregação é nula e a propagação degenera em `H ← αH`, ou seja, o vetor é multiplicado por exatamente α^hops.

| α | BP wFmax | BP Smin ↓ | CC wFmax | CC Smin ↓ | MF wFmax | MF Smin ↓ |
|---|---|---|---|---|---|---|
| *melhor hop* | h=1 |  | — |  | — |
| 0,1 | — | — | — | — | — | — |
| 0,3 | — | — | — | — | — | — |
| 0,5 | 0,5549 | 17,91 | — | — | — | — |
| 0,7 | — | — | — | — | — | — |
| 0,9 | — | — | — | — | — | — |

### Tabela 6.11 — GNN fim-a-fim: GCN vs GAT + MLP

Custo medido junto: a GNN treina o grafo a cada época, o SPMM fixo o pré-computa uma vez.


**VALIDAÇÃO**

| Ont | GNN | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ | h | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| BP | GCN | 0,6092 | 0,5775 | 0,5531 | 0,5997 | 0,6675 | 17,81 | — | — |
| BP | GAT | — | — | — | — | — | — | — | — |
| CC | GCN | 0,7782 | 0,7539 | 0,6577 | 0,7391 | 0,8367 | 8,66 | — | — |
| CC | GAT | — | — | — | — | — | — | — | — |
| MF | GCN | 0,7609 | 0,7191 | 0,6826 | 0,6420 | 0,8108 | 6,13 | — | — |
| MF | GAT | — | — | — | — | — | — | — | — |

**TESTE**

| Ont | GNN | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ | h | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| BP | GCN | 0,6158 | 0,5865 | 0,5643 | 0,6093 | 0,6766 | 18,16 | — | — |
| BP | GAT | — | — | — | — | — | — | — | — |
| CC | GCN | 0,7795 | 0,7554 | 0,6610 | 0,7441 | 0,8428 | 8,67 | — | — |
| CC | GAT | — | — | — | — | — | — | — | — |
| MF | GCN | 0,7607 | 0,7185 | 0,6794 | 0,6428 | 0,8110 | 6,07 | — | — |
| MF | GAT | — | — | — | — | — | — | — | — |

### Tabela 6.12 — Early Fusion: SPMM ⊕ GNN + MLP — VALIDAÇÃO

Com `concat` a entrada da MLP é 4096-d e com `add` é 3072-d, então `[1024, 512]` representa razões de compressão diferentes nos dois casos — a comparação entre fusões carrega essa diferença junto.

| Ont | Fusão | GNN | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|---|
| BP | `add` | GCN | — | — | — | — | — | — |
| BP | `add` | GAT | — | — | — | — | — | — |
| BP | `concat` | GCN | — | — | — | — | — | — |
| BP | `concat` | GAT | — | — | — | — | — | — |
| CC | `add` | GCN | — | — | — | — | — | — |
| CC | `add` | GAT | — | — | — | — | — | — |
| CC | `concat` | GCN | — | — | — | — | — | — |
| CC | `concat` | GAT | — | — | — | — | — | — |
| MF | `add` | GCN | — | — | — | — | — | — |
| MF | `add` | GAT | — | — | — | — | — | — |
| MF | `concat` | GCN | — | — | — | — | — | — |
| MF | `concat` | GAT | — | — | — | — | — | — |

### Tabela 6.14 — Hops com cabeça visual (grade grossa) — VALIDAÇÃO

Grade grossa de propósito: o objetivo é saber se o ótimo de hops se desloca ao trocar a cabeça, não remapear a curva inteira, que a Tabela 6.8 já tem com 11 pontos.

| hops | BP wFmax | BP Smin ↓ | CC wFmax | CC Smin ↓ | MF wFmax | MF Smin ↓ |
|---|---|---|---|---|---|---|

### Tabela 6.15 — α e GNN na rota visual — VALIDAÇÃO

| Ont | Configuração | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|

### Tabela 6.11b — GCN vs GAT em todas as famílias — VALIDAÇÃO

A comparação isolada da Tabela 6.11 cobre só a GNN fim-a-fim. Aqui ela atravessa as quatro combinações de método e cabeça, para verificar se a preferência por uma arquitetura se mantém ou se depende do contexto em que a GNN é usada.

| Método | Cabeça | Ont | GCN | GAT | Δ (GCN−GAT) |
|---|---|---|---|---|---|
| gnn | mlp | BP | 0,5531 | — | — |
| gnn | mlp | CC | 0,6577 | — | — |
| gnn | mlp | MF | 0,6826 | — | — |
| gnn | vision | BP | — | — | — |
| gnn | vision | CC | — | — | — |
| gnn | vision | MF | — | — | — |
| hybrid | mlp | BP | — | — | — |
| hybrid | mlp | CC | — | — | — |
| hybrid | mlp | MF | — | — | — |
| hybrid | vision | BP | — | — | — |
| hybrid | vision | CC | — | — | — |
| hybrid | vision | MF | — | — | — |

### Tabela 6.12b — Early Fusion: `add` vs `concat` — VALIDAÇÃO

Isola o efeito da operação de fusão. Lembrando que `concat` entrega 4096-d à MLP e `add` entrega 3072-d, então a comparação carrega junto uma diferença de compressão.

| Ont | GNN | `add` | `concat` | Δ (add−concat) |
|---|---|---|---|---|
| BP | GCN | — | — | — |
| BP | GAT | — | — | — |
| CC | GCN | — | — | — |
| CC | GAT | — | — | — |
| MF | GCN | — | — | — |
| MF | GAT | — | — | — |

### Tabela 6.16 — Custo computacional por família

O argumento de eficiência do trabalho depende desta tabela. Atenção: a família `hybrid` consome embeddings **pré-computados** pelo bloco `gnn`, então seu custo próprio não inclui o treino da GNN que o precede.

| Família | Runs | Horas (mediana) | Horas (total) | VRAM de pico | Épocas (mediana) |
|---|---|---|---|---|---|
| fixed + mlp | 3 | — | 0,0 | — GB | — |
| gnn + mlp | 3 | — | 0,0 | — GB | — |

**Total da campanha: 0,0 GPU-horas em 6 runs.**


### Verificação — concordância entre validação e teste

Se o teste fosse sistematicamente muito melhor que a validação, a seleção de hiperparâmetros estaria vazando para o split de teste.

Diferença teste − validação em wFmax, sobre as 6 runs: mediana **-0,0005**, mínimo -0,0083, máximo 0,0112, desvio 0,0068.


### Tabela 6.13 — Late Fusion: ensemble sobre as probabilidades

Combina as probabilidades já salvas das duas rotas campeãs, **sem treinar nada**. γ é o peso do SPMM: γ=1 é o SPMM puro, γ=0 é a GNN pura. Os parâmetros de cada estratégia são ajustados em **validação**; a coluna de teste é só reportada.

| Ont | Estratégia | wFmax val | wFmax teste | Parâmetros |
|---|---|---|---|---|
| BP | **linear** | **0,5730** | **0,5783** | gamma=0.55 |
| BP | max | 0,5668 | 0,5737 | — |
| BP | min | 0,5626 | 0,5697 | — |
| BP | ic_sigmoid | 0,5665 | 0,5714 | ic_thr=1.1, k=1.0 |
| BP | degree_sigmoid | 0,5729 | 0,5804 | d_thr=1.0, k=0.0 |
| BP | master | 0,5730 | 0,5797 | alpha=0.1, ic_thr=1.1, k_ic=1.0, d_thr=1.0, k_deg=0.0 |
| BP | stacking_dt | 0,5664 | 0,5724 | model=dt, max_depth=7 |
| BP | stacking_lr | 0,5715 | 0,5747 | model=lr, max_depth=7 |
| BP | stacking_mlp | 0,5654 | 0,5712 | params={"model": "mlp", "max_depth": 7, "feat_imp": None} |
| CC | **linear** | **0,6832** | **0,6846** | gamma=0.55 |
| CC | max | 0,6735 | 0,6749 | — |
| CC | min | 0,6767 | 0,6785 | — |
| CC | ic_sigmoid | 0,6776 | 0,6773 | ic_thr=0.69, k=0.5 |
| CC | degree_sigmoid | 0,6740 | 0,6749 | d_thr=17.0, k=0.25 |
| CC | master | 0,6791 | 0,6779 | alpha=0.7, ic_thr=0.69, k_ic=0.5, d_thr=17.0, k_deg=0.25 |
| CC | stacking_dt | 0,6760 | 0,6755 | model=dt, max_depth=7 |
| CC | stacking_lr | 0,6807 | 0,6797 | model=lr, max_depth=7 |
| CC | stacking_mlp | 0,6814 | 0,6807 | model=mlp, max_depth=7 |
| MF | **linear** | **0,7458** | **0,7407** | gamma=0.55 |
| MF | max | 0,7293 | 0,7229 | — |
| MF | min | 0,7280 | 0,7242 | — |
| MF | ic_sigmoid | 0,7409 | 0,7347 | ic_thr=0.62, k=0.5 |
| MF | degree_sigmoid | 0,7283 | 0,7213 | d_thr=30.0, k=0.25 |
| MF | master | 0,7414 | 0,7355 | alpha=0.9, ic_thr=0.62, k_ic=0.5, d_thr=30.0, k_deg=0.25 |
| MF | stacking_dt | 0,7388 | 0,7326 | model=dt, max_depth=7 |
| MF | stacking_lr | 0,7261 | 0,7244 | model=lr, max_depth=7 |
| MF | stacking_mlp | 0,7403 | 0,7339 | model=mlp, max_depth=7 |

### Tabela-síntese — melhor de cada rota, escolhido em VALIDAÇÃO, reportado em TESTE

O campeão de cada família é escolhido pelo wFmax de **validação**; as métricas exibidas são as de **teste**. Selecionar e reportar no mesmo split inflaria os números.

| Ont | Método | Cabeça | Config | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|---|---|
| BP | fixed | mlp | h=1 α=0,5 | 0,6112 | 0,5810 | 0,5570 | 0,6048 | 0,6602 | 18,30 |
| BP | fixed | vision | — | — | — | — | — | — | — |
| BP | gnn | mlp | GCN | 0,6158 | 0,5865 | 0,5643 | 0,6093 | 0,6766 | 18,16 |
| BP | gnn | vision | — | — | — | — | — | — | — |
| BP | hybrid | mlp | — | — | — | — | — | — | — |
| BP | hybrid | vision | — | — | — | — | — | — | — |
| CC | fixed | mlp | h=1 α=0,6 | 0,7818 | 0,7579 | 0,6685 | 0,7447 | 0,8414 | 8,81 |
| CC | fixed | vision | — | — | — | — | — | — | — |
| CC | gnn | mlp | GCN | 0,7795 | 0,7554 | 0,6610 | 0,7441 | 0,8428 | 8,67 |
| CC | gnn | vision | — | — | — | — | — | — | — |
| CC | hybrid | mlp | — | — | — | — | — | — | — |
| CC | hybrid | vision | — | — | — | — | — | — | — |
| MF | fixed | mlp | h=0 α=1,0 | 0,7979 | 0,7616 | 0,7262 | 0,6662 | 0,8325 | 5,29 |
| MF | fixed | vision | — | — | — | — | — | — | — |
| MF | gnn | mlp | GCN | 0,7607 | 0,7185 | 0,6794 | 0,6428 | 0,8110 | 6,07 |
| MF | gnn | vision | — | — | — | — | — | — | — |
| MF | hybrid | mlp | — | — | — | — | — | — | — |
| MF | hybrid | vision | — | — | — | — | — | — | — |

---

## Pendências

54 célula(s) sem run correspondente no CSV:

- `domain=bp, method=fixed, classifier_type=mlp, hops=0.0, alpha_init=0.5`
- `domain=bp, method=fixed, classifier_type=mlp, hops=1.0, alpha_init=0.1`
- `domain=bp, method=fixed, classifier_type=mlp, hops=1.0, alpha_init=0.3`
- `domain=bp, method=fixed, classifier_type=mlp, hops=1.0, alpha_init=0.7`
- `domain=bp, method=fixed, classifier_type=mlp, hops=1.0, alpha_init=0.9`
- `domain=bp, method=fixed, classifier_type=mlp, hops=10.0, alpha_init=0.5`
- `domain=bp, method=fixed, classifier_type=mlp, hops=2.0, alpha_init=0.5`
- `domain=bp, method=fixed, classifier_type=mlp, hops=3.0, alpha_init=0.5`
- `domain=bp, method=fixed, classifier_type=mlp, hops=4.0, alpha_init=0.5`
- `domain=bp, method=fixed, classifier_type=mlp, hops=5.0, alpha_init=0.5`
- `domain=bp, method=fixed, classifier_type=mlp, hops=6.0, alpha_init=0.5`
- `domain=bp, method=fixed, classifier_type=mlp, hops=7.0, alpha_init=0.5`
- `domain=bp, method=fixed, classifier_type=mlp, hops=8.0, alpha_init=0.5`
- `domain=bp, method=fixed, classifier_type=mlp, hops=9.0, alpha_init=0.5`
- `domain=bp, method=gnn, classifier_type=mlp, gnn_type=gat`
- `domain=bp, method=hybrid, classifier_type=mlp, hybrid_fusion=add, gnn_type=gat`
- `domain=bp, method=hybrid, classifier_type=mlp, hybrid_fusion=add, gnn_type=gcn`
- `domain=bp, method=hybrid, classifier_type=mlp, hybrid_fusion=concat, gnn_type=gat`
- `domain=bp, method=hybrid, classifier_type=mlp, hybrid_fusion=concat, gnn_type=gcn`
- `domain=cc, method=fixed, classifier_type=mlp, hops=0.0, alpha_init=0.5`
- `domain=cc, method=fixed, classifier_type=mlp, hops=1.0, alpha_init=0.5`
- `domain=cc, method=fixed, classifier_type=mlp, hops=10.0, alpha_init=0.5`
- `domain=cc, method=fixed, classifier_type=mlp, hops=2.0, alpha_init=0.5`
- `domain=cc, method=fixed, classifier_type=mlp, hops=3.0, alpha_init=0.5`
- `domain=cc, method=fixed, classifier_type=mlp, hops=4.0, alpha_init=0.5`
- `domain=cc, method=fixed, classifier_type=mlp, hops=5.0, alpha_init=0.5`
- `domain=cc, method=fixed, classifier_type=mlp, hops=6.0, alpha_init=0.5`
- `domain=cc, method=fixed, classifier_type=mlp, hops=7.0, alpha_init=0.5`
- `domain=cc, method=fixed, classifier_type=mlp, hops=8.0, alpha_init=0.5`
- `domain=cc, method=fixed, classifier_type=mlp, hops=9.0, alpha_init=0.5`
- `domain=cc, method=gnn, classifier_type=mlp, gnn_type=gat`
- `domain=cc, method=hybrid, classifier_type=mlp, hybrid_fusion=add, gnn_type=gat`
- `domain=cc, method=hybrid, classifier_type=mlp, hybrid_fusion=add, gnn_type=gcn`
- `domain=cc, method=hybrid, classifier_type=mlp, hybrid_fusion=concat, gnn_type=gat`
- `domain=cc, method=hybrid, classifier_type=mlp, hybrid_fusion=concat, gnn_type=gcn`
- `domain=mf, method=fixed, classifier_type=mlp, hops=0.0, alpha_init=0.5`
- `domain=mf, method=fixed, classifier_type=mlp, hops=1.0, alpha_init=0.5`
- `domain=mf, method=fixed, classifier_type=mlp, hops=10.0, alpha_init=0.5`
- `domain=mf, method=fixed, classifier_type=mlp, hops=2.0, alpha_init=0.5`
- `domain=mf, method=fixed, classifier_type=mlp, hops=3.0, alpha_init=0.5`
- ... e mais 11
