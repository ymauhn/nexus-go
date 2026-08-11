# Protocolo experimental e literatura na convenção `deepgraphgo`

As Tabelas 6.1–6.3 são **calculadas dos dados** por `scripts_v4/tabelas_protocolo.py`, não transcritas. As 6.16–6.18 vêm de `tabela_6metricas.json`.

---

## Tabela 6.1 — Escala dos dados

| | BP | CC | MF |
|---|---|---|---|
| Proteínas (treino / val / teste) | 73.768 / 9.221 / 9.221 | 74.328 / 9.292 / 9.292 | 62.909 / 7.864 / 7.864 |
| Nós no grafo PPI | 92.210 | 92.912 | 78.637 |
| Arestas não direcionadas | 492.920 | 566.566 | 470.206 |
| Proteínas isoladas no grafo | 36,65% | 34,95% | 32,19% |
| Termos GO | 500 | 498 | 499 |

Medido com `thr=0.1` e `dedup=max`, o mesmo grafo que os modelos consumiram. O grafo é construído sobre a união dos três splits, então o número de nós é a soma das três colunas de proteínas.


## Tabela 6.2 — Rede PPI: o artefato de multiplicidade

| | Medido |
|---|---|
| Linhas em `ppi.csv` | 2.422.234 |
| Pares distintos | 791.011 |
| Multiplicidade média (máxima) | 3,062 (170) |
| Pares com escore divergente entre repetições | **0** |
| Faixa de escores | [0,400; 0,999] |

**Nenhum par tem escores conflitantes** (0 divergências em 791.011 pares): a multiplicidade é artefato de montagem do arquivo, não informação. Somar produziria peso `c_ij · s_ij`, e como a multiplicidade varia entre vizinhos do mesmo nó, a normalização estocástica por coluna **não** cancela o efeito — a ordenação dos vizinhos se inverte. É a justificativa da campanha `dedup=max`.


## Tabela 6.3 — Configuração fixa

| Parâmetro | Campanha ProtT5+PPI | PPI-only |
|---|---|---|
| `dedup` | max | `max` (já era) |
| `batch_size` | 256 | 32 |
| `lr` | 0.0001 | 0.0001 |
| `max_epochs` | 250 | 200 |
| `patience` | 20 | 25 |
| `seed` | 1337 | 1337 |
| `mlp_dropout` | 0.4 | — |
| `thr` do grafo | 0.1 | — |

`thr` é **inerte**: o menor escore em `ppi.csv` é exatamente 0,400 e o filtro é `w < thr`, então nem 0,10 nem 0,40 removem aresta alguma. Isso elimina retroativamente o confundimento arquitetura × densidade entre GCN e GAT das tabelas antigas.

> **Nota metodológica.** `batch_size` difere entre as duas frentes e **não é apenas granularidade de gradiente**: em `ProteinLoss`, o termo `centric="go"` estima o F1 de cada termo GO *ao longo da dimensão do batch*. É hiperparâmetro da própria função de perda, e por isso os números das duas frentes não são comparáveis célula a célula.


---

# Literatura na convenção `deepgraphgo`

*Fonte:* `tabela_6metricas.json`, gerado em 2026-07-29T22:52:21-03:00. Definição: deepgraphgo/metrics.py — Smin em bits, Fmax* no τ ótimo do Fmax.

**Estes números não são comparáveis com os do confronto final** — outra convenção de métrica, com raiz removida e IC estimado da frequência de treino, o que põe o Smin na casa dos 64 bits em BP contra ~20 na convenção deste trabalho. Estão aqui para documentar o viés de cobertura.


## Tabela 6.16 — Conjunto de avaliação completo — TESTE

| Ont | Método | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ | Cob. |
|---|---|---|---|---|---|---|---|---|
| BP | DeepGraphGO | 0,5217 | 0,5310 | 0,4736 | 0,5568 | 0,5628 | 64,49 | 96,3% |
| BP | SEGT-GO | 0,4794 | 0,4894 | 0,4273 | 0,4935 | 0,5016 | 68,22 | 96,3% |
| BP | Mashup | 0,4194 | 0,5261 | 0,3802 | 0,3847 | 0,5514 | 74,59 | 63,6% |
| BP | MELISSA | 0,4016 | 0,5034 | 0,3606 | 0,3691 | 0,5354 | 75,80 | 63,6% |
| CC | DeepGraphGO | 0,7102 | 0,7286 | 0,5950 | 0,7875 | 0,8057 | 20,79 | 95,1% |
| CC | SEGT-GO | 0,6798 | 0,6974 | 0,5468 | 0,7465 | 0,7642 | 22,36 | 95,1% |
| CC | Mashup | 0,5744 | 0,7098 | 0,4778 | 0,5357 | 0,7028 | 23,68 | 66,6% |
| CC | MELISSA | 0,5651 | 0,6974 | 0,4660 | 0,5242 | 0,6919 | 24,10 | 66,6% |
| MF | DeepGraphGO | 0,7300 | 0,7469 | 0,6737 | 0,7836 | 0,7939 | 15,44 | 95,7% |
| MF | SEGT-GO | 0,6859 | 0,7013 | 0,6199 | 0,7163 | 0,7246 | 17,46 | 95,7% |
| MF | Mashup | 0,4987 | 0,6164 | 0,4195 | 0,4770 | 0,6229 | 23,58 | 68,7% |

## Tabela 6.18 — Restrito ao subconjunto da rede PPI — TESTE

| Ont | Método | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ | Cob. |
|---|---|---|---|---|---|---|---|---|
| BP | DeepGraphGO | 0,5515 | 0,5515 | 0,5033 | 0,6012 | 0,5909 | 66,66 | 100,0% |
| BP | Mashup | 0,5259 | 0,5261 | 0,4794 | 0,5791 | 0,5684 | 76,50 | 100,0% |
| BP | MELISSA | 0,5031 | 0,5034 | 0,4545 | 0,5546 | 0,5433 | 77,74 | 100,0% |
| BP | SEGT-GO | 0,4840 | 0,4852 | 0,4300 | 0,5059 | 0,4974 | 73,77 | 100,0% |
| CC | DeepGraphGO | 0,7354 | 0,7355 | 0,6205 | 0,8321 | 0,8258 | 20,73 | 100,0% |
| CC | Mashup | 0,7138 | 0,7146 | 0,5942 | 0,7940 | 0,7993 | 23,06 | 100,0% |
| CC | MELISSA | 0,7004 | 0,7013 | 0,5778 | 0,7766 | 0,7828 | 23,85 | 100,0% |
| CC | SEGT-GO | 0,7000 | 0,7000 | 0,5641 | 0,7860 | 0,7789 | 22,81 | 100,0% |
| MF | DeepGraphGO | 0,7537 | 0,7540 | 0,6939 | 0,8259 | 0,8155 | 16,35 | 100,0% |
| MF | SEGT-GO | 0,7033 | 0,7035 | 0,6318 | 0,7500 | 0,7370 | 18,83 | 100,0% |
| MF | Mashup | 0,6185 | 0,6214 | 0,5244 | 0,6887 | 0,6762 | 23,94 | 100,0% |

## Tabela 6.17 — O viés de cobertura

| | DeepGraphGO / SEGT-GO | Mashup / MELISSA |
|---|---|---|
| BP/teste — 9.221 proteínas no split | 96,3% (~8.880) | **63,6%** (~5.865) |
| CC/teste — 9.292 proteínas no split | 95,1% (~8.837) | **66,6%** (~6.188) |
| MF/teste — 7.864 proteínas no split | 95,7% (~7.526) | **68,7%** (~5.403) |
| Fora da rede PPI | predição por homologia (PSI-BLAST) | **descartadas** |

As contagens absolutas são aproximadas: o arquivo guarda a cobertura arredondada a três casas, não o número de proteínas de cada método.


`melissa_py/custom_data.py::load_custom_split` alinha as anotações ao universo de genes do PPI e descarta silenciosamente as proteínas ausentes do grafo. Os números do Mashup **não estavam errados** — foram reproduzidos exatamente — estavam medidos sobre um subconjunto mais fácil, o que invalidava a comparação. Daí a Tabela 6.18.


**Pendências neste arquivo:** MELISSA/mf/test/full, MELISSA/mf/valid/full, MELISSA/mf/test/subset, MELISSA/mf/valid/subset. Não afetam o confronto final, que usa a outra convenção.

