# Capítulo 6 — tabelas geradas

Gerado por `scripts_v4/tabelas_cap6.py` a partir de `campanha_dedupmax\tabelas\results_all.csv` (96 runs, `dedup=max`).
**Não editar à mão** — regenerar após novas runs. Célula `—` significa run ausente do CSV, nunca valor omitido.

Smin em bits, **menor é melhor**; nas demais métricas, maior é melhor.

### Tabela 6.8 — Ablação de hops, SPMM fixo + MLP (α=0,5) — VALIDAÇÃO

A curva completa de propagação. É a tabela que localiza o melhor número de saltos em cada ontologia, e o argumento de que o ganho satura cedo depende dela.

| hops | BP wFmax | BP Smin ↓ | CC wFmax | CC Smin ↓ | MF wFmax | MF Smin ↓ |
|---|---|---|---|---|---|---|
| 0 | 0,5336 | 18,51 | 0,6521 | 9,26 | 0,7318 | 5,32 |
| 1 | 0,5521 | 17,80 | 0,6611 | 8,96 | 0,7311 | 5,32 |
| 2 | 0,5406 | 18,06 | 0,6613 | 8,61 | 0,7118 | 5,67 |
| 3 | 0,5212 | 18,74 | 0,6471 | 9,04 | 0,6861 | 6,12 |
| 4 | 0,4985 | 19,55 | 0,6303 | 9,40 | 0,6670 | 6,44 |
| 5 | 0,4817 | 19,96 | 0,6158 | 9,71 | 0,6381 | 6,89 |
| 6 | 0,4645 | 20,50 | 0,5991 | 10,05 | 0,6075 | 7,39 |
| 7 | 0,4472 | 21,17 | 0,5686 | 10,44 | 0,5793 | 7,77 |
| 8 | 0,4208 | 21,58 | 0,5407 | 10,68 | 0,5681 | 7,92 |
| 9 | 0,4181 | 21,59 | 0,5364 | 10,75 | 0,5676 | 8,06 |
| 10 | 0,4161 | 21,75 | 0,5333 | 10,93 | 0,5558 | 8,02 |

### Tabela 6.10 — Ablação de α no melhor hop — VALIDAÇÃO

α controla quanto da representação própria do nó sobrevive a cada salto. Para um nó isolado a agregação é nula e a propagação degenera em `H ← αH`, ou seja, o vetor é multiplicado por exatamente α^hops.

| α | BP wFmax | BP Smin ↓ | CC wFmax | CC Smin ↓ | MF wFmax | MF Smin ↓ |
|---|---|---|---|---|---|---|
| *melhor hop* | h=1 |  | h=2 |  | h=0 |
| 0,1 | 0,4972 | 19,36 | 0,5849 | 10,23 | n/a | n/a |
| 0,3 | 0,5346 | 18,30 | 0,6360 | 9,29 | n/a | n/a |
| 0,5 | 0,5521 | 17,80 | 0,6613 | 8,61 | 0,7318 | 5,32 |
| 0,7 | 0,5523 | 17,89 | 0,6684 | 8,69 | n/a | n/a |
| 0,9 | 0,5338 | 18,56 | 0,6631 | 8,92 | n/a | n/a |

`n/a` em MF: o melhor hop é **0**, e com zero saltos o SPMM é a identidade — α não participa de nenhuma operação. A varredura foi pulada por isso, não por falta de execução.


### Tabela 6.11 — GNN fim-a-fim: GCN vs GAT + MLP

Custo medido junto: a GNN treina o grafo a cada época, o SPMM fixo o pré-computa uma vez.


**VALIDAÇÃO**

| Ont | GNN | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ | h | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| BP | GCN | 0,6127 | 0,5812 | 0,5578 | 0,5988 | 0,6528 | 17,39 | 4,77 | 12,6 GB |
| BP | GAT | 0,5308 | 0,4947 | 0,4647 | 0,5025 | 0,5715 | 20,96 | 7,40 | 18,4 GB |
| CC | GCN | 0,7821 | 0,7580 | 0,6635 | 0,7380 | 0,8322 | 8,50 | 4,20 | 14,1 GB |
| CC | GAT | 0,7490 | 0,7211 | 0,6114 | 0,7087 | 0,8119 | 9,85 | 11,51 | 20,7 GB |
| MF | GCN | 0,7609 | 0,7172 | 0,6783 | 0,6738 | 0,8065 | 6,09 | 1,93 | 11,8 GB |
| MF | GAT | 0,7278 | 0,6803 | 0,6383 | 0,5975 | 0,7595 | 6,94 | 8,85 | 17,3 GB |

**TESTE**

| Ont | GNN | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ | h | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| BP | GCN | 0,6205 | 0,5917 | 0,5695 | 0,6097 | 0,6627 | 17,66 | 4,77 | 12,6 GB |
| BP | GAT | 0,5315 | 0,4971 | 0,4673 | 0,5067 | 0,5752 | 21,52 | 7,40 | 18,4 GB |
| CC | GCN | 0,7842 | 0,7601 | 0,6670 | 0,7441 | 0,8407 | 8,44 | 4,20 | 14,1 GB |
| CC | GAT | 0,7477 | 0,7199 | 0,6105 | 0,7103 | 0,8119 | 9,93 | 11,51 | 20,7 GB |
| MF | GCN | 0,7609 | 0,7174 | 0,6777 | 0,6750 | 0,8064 | 6,04 | 1,93 | 11,8 GB |
| MF | GAT | 0,7253 | 0,6763 | 0,6319 | 0,5964 | 0,7538 | 7,00 | 8,85 | 17,3 GB |

### Tabela 6.12 — Early Fusion: SPMM ⊕ GNN + MLP — VALIDAÇÃO

Com `concat` a entrada da MLP é 4096-d e com `add` é 3072-d, então `[1024, 512]` representa razões de compressão diferentes nos dois casos — a comparação entre fusões carrega essa diferença junto.

| Ont | Fusão | GNN | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|---|
| BP | `add` | GCN | 0,6215 | 0,5893 | 0,5652 | 0,6137 | 0,6468 | 17,01 |
| BP | `add` | GAT | 0,5778 | 0,5434 | 0,5152 | 0,5726 | 0,6187 | 19,10 |
| BP | `concat` | GCN | 0,6186 | 0,5853 | 0,5605 | 0,6284 | 0,6396 | 17,23 |
| BP | `concat` | GAT | 0,5899 | 0,5567 | 0,5299 | 0,5800 | 0,6267 | 18,73 |
| CC | `add` | GCN | 0,7855 | 0,7605 | 0,6643 | 0,8499 | 0,8277 | 8,47 |
| CC | `add` | GAT | 0,7704 | 0,7449 | 0,6478 | 0,7264 | 0,8215 | 9,02 |
| CC | `concat` | GCN | 0,7866 | 0,7623 | 0,6706 | 0,8210 | 0,8283 | 8,34 |
| CC | `concat` | GAT | 0,7763 | 0,7518 | 0,6567 | 0,7525 | 0,8235 | 8,83 |
| MF | `add` | GCN | 0,8064 | 0,7710 | 0,7381 | 0,8228 | 0,8254 | 5,18 |
| MF | `add` | GAT | 0,7801 | 0,7404 | 0,7057 | 0,6532 | 0,8173 | 5,68 |
| MF | `concat` | GCN | 0,8052 | 0,7697 | 0,7368 | 0,8419 | 0,8250 | 5,20 |
| MF | `concat` | GAT | 0,7923 | 0,7545 | 0,7209 | 0,6858 | 0,8195 | 5,42 |

### Tabela 6.14 — Hops com cabeça visual (grade grossa) — VALIDAÇÃO

Grade grossa de propósito: o objetivo é saber se o ótimo de hops se desloca ao trocar a cabeça, não remapear a curva inteira, que a Tabela 6.8 já tem com 11 pontos.

| hops | BP wFmax | BP Smin ↓ | CC wFmax | CC Smin ↓ | MF wFmax | MF Smin ↓ |
|---|---|---|---|---|---|---|
| 0 | 0,4633 | 20,49 | 0,6013 | 10,21 | 0,6825 | 6,11 |
| 1 | 0,4719 | 20,22 | 0,6104 | 10,00 | 0,6704 | 6,24 |
| 2 | 0,4874 | 19,60 | 0,6191 | 9,65 | 0,6638 | 6,33 |
| 3 | 0,4978 | 19,31 | 0,6253 | 9,44 | 0,6538 | 6,38 |
| 5 | 0,5177 | 18,58 | 0,6283 | 9,19 | 0,6350 | 6,69 |
| 7 | 0,5217 | 18,46 | 0,6247 | 9,35 | 0,6317 | 6,78 |
| 10 | 0,5098 | 19,11 | 0,6212 | 9,54 | 0,6274 | 6,90 |

### Tabela 6.15 — α e GNN na rota visual — VALIDAÇÃO

| Ont | Configuração | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| BP | fixo h=0 α=0,5 | 0,5347 | 0,4962 | 0,4633 | 0,4605 | 0,5302 | 20,49 |
| BP | fixo h=1 α=0,5 | 0,5407 | 0,5033 | 0,4719 | 0,4855 | 0,5488 | 20,22 |
| BP | fixo h=2 α=0,5 | 0,5543 | 0,5168 | 0,4874 | 0,4892 | 0,5631 | 19,60 |
| BP | fixo h=3 α=0,5 | 0,5628 | 0,5256 | 0,4978 | 0,4975 | 0,5657 | 19,31 |
| BP | fixo h=5 α=0,5 | 0,5790 | 0,5442 | 0,5177 | 0,4752 | 0,5652 | 18,58 |
| BP | fixo h=7 α=0,3 | 0,5649 | 0,5286 | 0,5015 | 0,6003 | 0,5620 | 19,39 |
| BP | fixo h=7 α=0,5 | 0,5830 | 0,5478 | 0,5217 | 0,6175 | 0,5688 | 18,46 |
| BP | fixo h=7 α=0,7 | 0,5740 | 0,5373 | 0,5102 | 0,4854 | 0,5693 | 18,72 |
| BP | fixo h=10 α=0,5 | 0,5710 | 0,5365 | 0,5098 | 0,5034 | 0,5806 | 19,11 |
| BP | GNN GAT | 0,5279 | 0,4913 | 0,4617 | 0,5092 | 0,5681 | 20,56 |
| BP | GNN GCN | 0,5896 | 0,5568 | 0,5310 | 0,5641 | 0,5987 | 18,29 |
| BP | híbrido `add` GAT | 0,5069 | 0,4677 | 0,4374 | 0,4671 | 0,5155 | 21,60 |
| BP | híbrido `add` GCN | 0,5668 | 0,5308 | 0,5021 | 0,5058 | 0,5784 | 19,16 |
| CC | fixo h=0 α=0,5 | 0,7420 | 0,7129 | 0,6013 | 0,6068 | 0,7678 | 10,21 |
| CC | fixo h=1 α=0,5 | 0,7464 | 0,7182 | 0,6104 | 0,6166 | 0,7810 | 10,00 |
| CC | fixo h=2 α=0,5 | 0,7546 | 0,7271 | 0,6191 | 0,6259 | 0,7930 | 9,65 |
| CC | fixo h=3 α=0,5 | 0,7576 | 0,7305 | 0,6253 | 0,6085 | 0,7932 | 9,44 |
| CC | fixo h=5 α=0,3 | 0,7529 | 0,7254 | 0,6176 | 0,6351 | 0,7991 | 9,59 |
| CC | fixo h=5 α=0,5 | 0,7612 | 0,7340 | 0,6283 | 0,6007 | 0,7876 | 9,19 |
| CC | fixo h=5 α=0,7 | 0,7584 | 0,7305 | 0,6235 | 0,6103 | 0,7944 | 9,49 |
| CC | fixo h=7 α=0,5 | 0,7589 | 0,7322 | 0,6247 | 0,6137 | 0,7885 | 9,35 |
| CC | fixo h=10 α=0,5 | 0,7561 | 0,7287 | 0,6212 | 0,6299 | 0,7969 | 9,54 |
| CC | GNN GAT | 0,7393 | 0,7104 | 0,5945 | 0,6935 | 0,7899 | 10,09 |
| CC | GNN GCN | 0,7621 | 0,7363 | 0,6333 | 0,7376 | 0,8099 | 9,33 |
| CC | híbrido `add` GAT | 0,7280 | 0,6971 | 0,5765 | 0,6404 | 0,7736 | 10,38 |
| CC | híbrido `add` GCN | 0,7585 | 0,7315 | 0,6256 | 0,6471 | 0,8043 | 9,40 |
| MF | fixo h=0 α=0,5 | 0,7642 | 0,7216 | 0,6825 | 0,5876 | 0,7699 | 6,11 |
| MF | fixo h=1 α=0,5 | 0,7562 | 0,7105 | 0,6704 | 0,6112 | 0,7767 | 6,24 |
| MF | fixo h=2 α=0,5 | 0,7502 | 0,7045 | 0,6638 | 0,5766 | 0,7542 | 6,33 |
| MF | fixo h=3 α=0,5 | 0,7430 | 0,6939 | 0,6538 | 0,5757 | 0,7482 | 6,38 |
| MF | fixo h=5 α=0,5 | 0,7283 | 0,6768 | 0,6350 | 0,5854 | 0,7424 | 6,69 |
| MF | fixo h=7 α=0,5 | 0,7259 | 0,6754 | 0,6317 | 0,5740 | 0,7379 | 6,78 |
| MF | fixo h=10 α=0,5 | 0,7206 | 0,6702 | 0,6274 | 0,5762 | 0,7285 | 6,90 |
| MF | GNN GAT | 0,7092 | 0,6575 | 0,6103 | 0,5736 | 0,7248 | 7,34 |
| MF | GNN GCN | 0,7417 | 0,6956 | 0,6546 | 0,6894 | 0,7711 | 6,54 |
| MF | híbrido `add` GAT | 0,7009 | 0,6434 | 0,5944 | 0,5689 | 0,6988 | 7,37 |
| MF | híbrido `add` GCN | 0,7601 | 0,7161 | 0,6766 | 0,5796 | 0,7675 | 6,12 |

### Tabela 6.11b — GCN vs GAT em todas as famílias — VALIDAÇÃO

A comparação isolada da Tabela 6.11 cobre só a GNN fim-a-fim. Aqui ela atravessa as quatro combinações de método e cabeça, para verificar se a preferência por uma arquitetura se mantém ou se depende do contexto em que a GNN é usada.

| Método | Cabeça | Ont | GCN | GAT | Δ (GCN−GAT) |
|---|---|---|---|---|---|
| gnn | mlp | BP | **0,5578** | 0,4647 | **+0,0931** |
| gnn | mlp | CC | **0,6635** | 0,6114 | **+0,0521** |
| gnn | mlp | MF | **0,6783** | 0,6383 | **+0,0400** |
| gnn | vision | BP | **0,5310** | 0,4617 | **+0,0694** |
| gnn | vision | CC | **0,6333** | 0,5945 | **+0,0388** |
| gnn | vision | MF | **0,6546** | 0,6103 | **+0,0444** |
| hybrid | mlp | BP | **0,5652** | 0,5299 | **+0,0352** |
| hybrid | mlp | CC | **0,6706** | 0,6567 | **+0,0139** |
| hybrid | mlp | MF | **0,7381** | 0,7209 | **+0,0173** |
| hybrid | vision | BP | **0,5021** | 0,4374 | **+0,0646** |
| hybrid | vision | CC | **0,6256** | 0,5765 | **+0,0490** |
| hybrid | vision | MF | **0,6766** | 0,5944 | **+0,0822** |

A GCN vence em **12 de 12** comparações.


### Tabela 6.12b — Early Fusion: `add` vs `concat` — VALIDAÇÃO

Isola o efeito da operação de fusão. Lembrando que `concat` entrega 4096-d à MLP e `add` entrega 3072-d, então a comparação carrega junto uma diferença de compressão.

| Ont | GNN | `add` | `concat` | Δ (add−concat) |
|---|---|---|---|---|
| BP | GCN | 0,5652 | 0,5605 | +0,0047 |
| BP | GAT | 0,5152 | 0,5299 | -0,0147 |
| CC | GCN | 0,6643 | 0,6706 | -0,0063 |
| CC | GAT | 0,6478 | 0,6567 | -0,0089 |
| MF | GCN | 0,7381 | 0,7368 | +0,0013 |
| MF | GAT | 0,7057 | 0,7209 | -0,0151 |

### Tabela 6.16 — Custo computacional por família

O argumento de eficiência do trabalho depende desta tabela. Atenção: a família `hybrid` consome embeddings **pré-computados** pelo bloco `gnn`, então seu custo próprio não inclui o treino da GNN que o precede.

| Família | Runs | Horas (mediana) | Horas (total) | VRAM de pico | Épocas (mediana) |
|---|---|---|---|---|---|
| fixed + mlp | 41 | 0,32 | 12,5 | 6,4 GB | 243 |
| fixed + vision | 25 | 1,39 | 36,2 | 6,4 GB | 34 |
| gnn + mlp | 6 | 6,09 | 38,7 | 20,7 GB | 213 |
| gnn + vision | 6 | 4,35 | 31,3 | 21,1 GB | 138 |
| hybrid + mlp | 12 | 0,14 | 1,8 | 6,4 GB | 112 |
| hybrid + vision | 6 | 0,17 | 1,0 | 6,4 GB | 32 |

**Total da campanha: 121,5 GPU-horas em 96 runs.**


### Verificação — concordância entre validação e teste

Se o teste fosse sistematicamente muito melhor que a validação, a seleção de hiperparâmetros estaria vazando para o split de teste.

Diferença teste − validação em wFmax, sobre as 96 runs: mediana **0,0026**, mínimo -0,0102, máximo 0,0117, desvio 0,0056.


### Tabela 6.13 — Late Fusion: ensemble sobre as probabilidades

Combina as probabilidades já salvas das duas rotas campeãs, **sem treinar nada**. γ é o peso do SPMM: γ=1 é o SPMM puro, γ=0 é a GNN pura. Os parâmetros de cada estratégia são ajustados em **validação**; a coluna de teste é só reportada.

| Ont | Estratégia | wFmax val | wFmax teste | Parâmetros |
|---|---|---|---|---|
| BP | **linear** | **0,5796** | **0,5869** | gamma=0.5 |
| BP | max | 0,5735 | 0,5805 | — |
| BP | min | 0,5638 | 0,5721 | — |
| BP | ic_sigmoid | 0,5703 | 0,5745 | ic_thr=1.1, k=1.0 |
| BP | degree_sigmoid | 0,5702 | 0,5777 | d_thr=3.0, k=0.25 |
| BP | master | 0,5742 | 0,5807 | alpha=0.5, ic_thr=1.1, k_ic=1.0, d_thr=3.0, k_deg=0.25 |
| BP | stacking_dt | 0,5622 | 0,5701 | model=dt, max_depth=7 |
| BP | stacking_lr | 0,5753 | 0,5816 | model=lr, max_depth=7 |
| BP | stacking_mlp | 0,5692 | 0,5774 | model=mlp, max_depth=7 |
| BP | stacking_rf | 0,5665 | 0,5746 | model=rf, max_depth=7 |
| BP | stacking_gb | 0,5667 | 0,5740 | model=gb, max_depth=7 |
| CC | **linear** | **0,6849** | **0,6865** | gamma=0.55 |
| CC | max | 0,6757 | 0,6789 | — |
| CC | min | 0,6784 | 0,6826 | — |
| CC | ic_sigmoid | 0,6733 | 0,6754 | ic_thr=0.4, k=1.0 |
| CC | degree_sigmoid | 0,6726 | 0,6762 | d_thr=14.0, k=0.25 |
| CC | master | 0,6775 | 0,6804 | alpha=0.5, ic_thr=0.4, k_ic=1.0, d_thr=14.0, k_deg=0.25 |
| CC | stacking_dt | 0,6777 | 0,6783 | model=dt, max_depth=7 |
| CC | stacking_lr | 0,6807 | 0,6841 | model=lr, max_depth=7 |
| CC | stacking_mlp | 0,6808 | 0,6830 | model=mlp, max_depth=7 |
| CC | stacking_rf | 0,6810 | 0,6827 | model=rf, max_depth=7 |
| CC | stacking_gb | 0,6685 | 0,6700 | model=gb, max_depth=7 |
| MF | **linear** | **0,7444** | **0,7389** | gamma=0.55 |
| MF | max | 0,7259 | 0,7209 | — |
| MF | min | 0,7269 | 0,7246 | — |
| MF | ic_sigmoid | 0,7388 | 0,7322 | ic_thr=0.4, k=1.0 |
| MF | degree_sigmoid | 0,7242 | 0,7197 | d_thr=28.0, k=0.25 |
| MF | master | 0,7393 | 0,7327 | alpha=0.9, ic_thr=0.4, k_ic=1.0, d_thr=28.0, k_deg=0.25 |
| MF | stacking_dt | 0,7376 | 0,7294 | model=dt, max_depth=7 |
| MF | stacking_lr | 0,7247 | 0,7197 | model=lr, max_depth=7 |
| MF | stacking_mlp | 0,7392 | 0,7316 | model=mlp, max_depth=7 |
| MF | stacking_rf | 0,7414 | 0,7347 | model=rf, max_depth=7 |
| MF | stacking_gb | 0,7418 | 0,7338 | model=gb, max_depth=7 |

### Tabela-síntese — melhor de cada rota, escolhido em VALIDAÇÃO, reportado em TESTE

O campeão de cada família é escolhido pelo wFmax de **validação**; as métricas exibidas são as de **teste**. Selecionar e reportar no mesmo split inflaria os números.

| Ont | Método | Cabeça | Config | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|---|---|
| BP | fixed | mlp | h=1 α=0,7 | 0,6127 | 0,5813 | 0,5561 | 0,6039 | 0,6509 | 18,30 |
| BP | fixed | vision | h=7 α=0,5 | 0,5907 | 0,5581 | 0,5322 | 0,6241 | 0,5752 | 18,85 |
| BP | gnn | mlp | h=0 α=0,0 GCN | 0,6205 | 0,5917 | 0,5695 | 0,6097 | 0,6627 | 17,66 |
| BP | gnn | vision | h=0 α=0,0 GCN | 0,5945 | 0,5632 | 0,5383 | 0,5709 | 0,6048 | 18,80 |
| BP | hybrid | mlp | h=1 α=0,7 GCN `add` | 0,6287 | 0,5981 | 0,5751 | 0,6242 | 0,6573 | 17,35 |
| BP | hybrid | vision | h=7 α=0,5 GCN `add` | 0,5686 | 0,5339 | 0,5052 | 0,5101 | 0,5816 | 19,72 |
| CC | fixed | mlp | h=2 α=0,7 | 0,7816 | 0,7586 | 0,6700 | 0,7427 | 0,8388 | 8,76 |
| CC | fixed | vision | h=5 α=0,5 | 0,7607 | 0,7340 | 0,6293 | 0,6040 | 0,7872 | 9,25 |
| CC | gnn | mlp | h=0 α=0,0 GCN | 0,7842 | 0,7601 | 0,6670 | 0,7441 | 0,8407 | 8,44 |
| CC | gnn | vision | h=0 α=0,0 GCN | 0,7625 | 0,7364 | 0,6340 | 0,7413 | 0,8114 | 9,32 |
| CC | hybrid | mlp | h=2 α=0,7 GCN `concat` | 0,7882 | 0,7644 | 0,6747 | 0,8269 | 0,8321 | 8,34 |
| CC | hybrid | vision | h=5 α=0,5 GCN `add` | 0,7578 | 0,7308 | 0,6263 | 0,6487 | 0,8052 | 9,49 |
| MF | fixed | mlp | h=0 α=0,5 | 0,7975 | 0,7602 | 0,7257 | 0,7375 | 0,8225 | 5,31 |
| MF | fixed | vision | h=0 α=0,5 | 0,7652 | 0,7211 | 0,6804 | 0,5901 | 0,7707 | 6,02 |
| MF | gnn | mlp | h=0 α=0,0 GCN | 0,7609 | 0,7174 | 0,6777 | 0,6750 | 0,8064 | 6,04 |
| MF | gnn | vision | h=0 α=0,0 GCN | 0,7396 | 0,6938 | 0,6508 | 0,6878 | 0,7693 | 6,58 |
| MF | hybrid | mlp | h=0 α=0,5 GCN `add` | 0,8022 | 0,7654 | 0,7294 | 0,8188 | 0,8205 | 5,22 |
| MF | hybrid | vision | h=0 α=0,5 GCN `add` | 0,7587 | 0,7139 | 0,6732 | 0,5831 | 0,7658 | 6,10 |

### Tabela 6.7 — Ablação C do PPI-only: cabeças de imagem — TESTE

Sem ProtT5: o único sinal é a topologia. Serve de piso para medir quanto o ProtT5 acrescenta nas tabelas acima.

| Ont | Backbone | Pool | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|---|
| BP | ConvNeXt-Tiny | `max` | 0,3793 | 0,3286 | 0,3033 | 0,3890 | 0,3899 | 25,80 |
| BP | ConvNeXt-Tiny | `avg` | 0,3844 | 0,3312 | 0,3076 | 0,3990 | 0,3959 | 25,93 |
| BP | ResNet50 | `max` | 0,3760 | 0,3285 | 0,2974 | 0,3819 | 0,3897 | 25,65 |
| BP | ResNet50 | `avg` | 0,3729 | 0,3215 | 0,2968 | 0,3764 | 0,3844 | 26,42 |
| CC | ConvNeXt-Tiny | `max` | 0,6399 | 0,5955 | 0,4095 | 0,6805 | 0,6760 | 12,51 |
| CC | ConvNeXt-Tiny | `avg` | 0,6408 | 0,5957 | 0,4051 | 0,6817 | 0,6737 | 12,49 |
| CC | ResNet50 | `max` | 0,6380 | 0,5913 | 0,4045 | 0,6621 | 0,6755 | 12,55 |
| CC | ResNet50 | `avg` | 0,6329 | 0,5847 | 0,4015 | 0,6545 | 0,6692 | 12,67 |
| MF | ConvNeXt-Tiny | `max` | 0,5808 | 0,4917 | 0,4097 | 0,5362 | 0,5734 | 9,15 |
| MF | ConvNeXt-Tiny | `avg` | 0,5828 | 0,4939 | 0,4149 | 0,5430 | 0,5795 | 9,19 |
| MF | ResNet50 | `max` | 0,5813 | 0,4901 | 0,4092 | 0,5300 | 0,5793 | 9,30 |
| MF | ResNet50 | `avg` | 0,5875 | 0,4997 | 0,4186 | 0,5774 | 0,5800 | 9,53 |

---

## Pendências

Nenhuma — todas as células têm run correspondente.
