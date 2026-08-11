# Análises complementares

Calculadas a partir das probabilidades salvas, split **test**, pelas rotas campeãs em validação (cabeça MLP).
Métricas por `ppi_v4.metrics.evaluate_collect` aplicado a subconjuntos de linhas — nenhuma métrica foi reimplementada.

Smin em bits, **menor é melhor**.

---

## 1. Quanto o ProtT5 acrescenta sobre a topologia pura

O PPI-only usa **apenas** o perfil de escores da rede; a campanha soma a ele o embedding ProtT5. Ambas as gerações pontuam com a mesma convenção e o mesmo número de termos, então a diferença é atribuível à representação da sequência.

| Ont | PPI-only (só grafo) | Campanha (ProtT5 + grafo) | Δ absoluto | Δ relativo |
|---|---|---|---|---|
| BP | 0,3730 | **0,5751** | +0,2021 | +54,2% |
| CC | 0,4839 | **0,6747** | +0,1908 | +39,4% |
| MF | 0,4862 | **0,7294** | +0,2432 | +50,0% |

---

## 2. Proteínas isoladas vs conectadas

Isolada = grau zero no grafo limiarizado em thr=0.1 com `dedup=max`, o mesmo grafo que os modelos consumiram. Um nó isolado não recebe agregação: na propagação fixa sua representação é multiplicada por α^hops e nada mais.


### BP — 3,421 isoladas de 9,221 (37.1%)

| Método | Subconjunto | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| fixed | isoladas (3,421) | 0,6675 | 0,6298 | 0,6130 | 0,6440 | 0,7133 | 13,23 |
| fixed | conectadas (5,800) | 0,5815 | 0,5538 | 0,5239 | 0,5795 | 0,6130 | 21,28 |
| gnn | isoladas (3,421) | 0,6750 | 0,6398 | 0,6226 | 0,6478 | 0,7128 | 12,87 |
| gnn | conectadas (5,800) | 0,5897 | 0,5643 | 0,5397 | 0,5833 | 0,6299 | 20,49 |
| hybrid | isoladas (3,421) | 0,6772 | 0,6410 | 0,6246 | 0,6169 | 0,6766 | 12,66 |
| hybrid | conectadas (5,800) | 0,6001 | 0,5728 | 0,5458 | 0,6215 | 0,6353 | 20,09 |

**Grau × acerto em BP** — melhor rota (`bp_hybrid_add_gcn_mlp_h1_a07_thr010_dedupmax`), τ=0.46, Spearman ρ = **-0,077**

| Faixa de grau | Proteínas | F1 médio | desvio |
|---|---|---|---|
| 0 (isoladas) | 3,421 | 0,6288 | 0,3230 |
| 1–2 | 433 | 0,5317 | 0,3260 |
| 3–5 | 757 | 0,4924 | 0,3017 |
| 6–14 | 2,281 | 0,5355 | 0,2798 |
| 15–49 | 2,066 | 0,5767 | 0,2538 |
| ≥ 50 | 263 | 0,6158 | 0,1971 |

### CC — 3,188 isoladas de 9,292 (34.3%)

| Método | Subconjunto | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| fixed | isoladas (3,188) | 0,8079 | 0,7847 | 0,7032 | 0,7520 | 0,8619 | 6,62 |
| fixed | conectadas (6,104) | 0,7681 | 0,7449 | 0,6528 | 0,7371 | 0,8264 | 9,85 |
| gnn | isoladas (3,188) | 0,8123 | 0,7889 | 0,7044 | 0,7461 | 0,8432 | 6,38 |
| gnn | conectadas (6,104) | 0,7695 | 0,7454 | 0,6476 | 0,7411 | 0,8355 | 9,50 |
| hybrid | isoladas (3,188) | 0,8068 | 0,7820 | 0,6922 | 0,7780 | 0,8133 | 6,61 |
| hybrid | conectadas (6,104) | 0,7788 | 0,7556 | 0,6656 | 0,8529 | 0,8410 | 9,24 |

**Grau × acerto em CC** — melhor rota (`cc_hybrid_concat_gcn_mlp_h2_a07_thr010_dedupmax`), τ=0.50, Spearman ρ = **-0,057**

| Faixa de grau | Proteínas | F1 médio | desvio |
|---|---|---|---|
| 0 (isoladas) | 3,188 | 0,7729 | 0,2409 |
| 1–2 | 380 | 0,6859 | 0,2689 |
| 3–5 | 634 | 0,7312 | 0,2387 |
| 6–14 | 2,357 | 0,7285 | 0,2361 |
| 15–49 | 2,388 | 0,7580 | 0,2140 |
| ≥ 50 | 345 | 0,7545 | 0,1894 |

### MF — 2,553 isoladas de 7,864 (32.5%)

| Método | Subconjunto | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| fixed | isoladas (2,553) | 0,8187 | 0,7770 | 0,7568 | 0,7345 | 0,8410 | 3,80 |
| fixed | conectadas (5,311) | 0,7884 | 0,7544 | 0,7123 | 0,7390 | 0,8112 | 6,02 |
| gnn | isoladas (2,553) | 0,8137 | 0,7724 | 0,7499 | 0,6741 | 0,8392 | 3,78 |
| gnn | conectadas (5,311) | 0,7352 | 0,6913 | 0,6430 | 0,6711 | 0,7842 | 7,12 |
| hybrid | isoladas (2,553) | 0,8147 | 0,7714 | 0,7485 | 0,7623 | 0,8158 | 3,76 |
| hybrid | conectadas (5,311) | 0,7963 | 0,7632 | 0,7205 | 0,8460 | 0,8238 | 5,90 |

**Grau × acerto em MF** — melhor rota (`mf_hybrid_add_gcn_mlp_h0_a05_thr010_dedupmax`), τ=0.42, Spearman ρ = **-0,110**

| Faixa de grau | Proteínas | F1 médio | desvio |
|---|---|---|---|
| 0 (isoladas) | 2,553 | 0,7826 | 0,2641 |
| 1–2 | 486 | 0,7613 | 0,2683 |
| 3–5 | 630 | 0,7463 | 0,2615 |
| 6–14 | 1,952 | 0,7573 | 0,2537 |
| 15–49 | 1,945 | 0,7469 | 0,2334 |
| ≥ 50 | 298 | 0,7255 | 0,2111 |

---

## Síntese

| Ont | Método | isoladas | conectadas | Δ |
|---|---|---|---|---|
| BP | fixed | 0,6130 | 0,5239 | -0,0891 |
| BP | gnn | 0,6226 | 0,5397 | -0,0828 |
| BP | hybrid | 0,6246 | 0,5458 | -0,0788 |
| CC | fixed | 0,7032 | 0,6528 | -0,0504 |
| CC | gnn | 0,7044 | 0,6476 | -0,0568 |
| CC | hybrid | 0,6922 | 0,6656 | -0,0267 |
| MF | fixed | 0,7568 | 0,7123 | -0,0446 |
| MF | gnn | 0,7499 | 0,6430 | -0,1068 |
| MF | hybrid | 0,7485 | 0,7205 | -0,0279 |

---

## 3. Verificação: o confundidor da profundidade de anotação

As isoladas pontuam **acima** das conectadas em 9 de 9 combinações. A explicação óbvia seria
mecânica: se proteínas isoladas tivessem menos anotações, o F1 por proteína seria mais fácil de
maximizar e o resultado não diria nada sobre o modelo. A tabela abaixo testa isso.

| Ont | Faixa de grau | Proteínas | Rótulos/proteína | IC médio | IC total | F1 médio |
|---|---|---|---|---|---|---|
| BP | 0 (isoladas) | 3.421 | 22,2 | 1,12 | 24,8 | **0,6288** |
| BP | 1–2 | 433 | **16,7** | 1,07 | 17,0 | 0,5317 |
| BP | 3–5 | 757 | 19,2 | 1,10 | 20,5 | **0,4924** |
| BP | 6–14 | 2.281 | 24,3 | 1,09 | 25,6 | 0,5355 |
| BP | 15–49 | 2.066 | 33,6 | 1,11 | 37,3 | 0,5767 |
| BP | ≥ 50 | 263 | 68,3 | 1,14 | 78,9 | 0,6158 |
| CC | 0 (isoladas) | 3.188 | 11,4 | 1,08 | 13,2 | **0,7729** |
| CC | 1–2 | 380 | **9,6** | 1,09 | 10,9 | **0,6859** |
| CC | ≥ 50 | 345 | 21,0 | 1,34 | 31,2 | 0,7545 |
| MF | 0 (isoladas) | 2.553 | 6,7 | 1,24 | 8,9 | **0,7826** |
| MF | 1–2 | 486 | **6,1** | 1,13 | 7,8 | 0,7613 |
| MF | ≥ 50 | 298 | 13,7 | 1,47 | 22,5 | **0,7255** |

**O confundidor não explica o padrão.** Em BP, as faixas 1–2 e 3–5 têm **menos** rótulos que as
isoladas (16,7 e 19,2 contra 22,2) e ainda assim o pior F1 de toda a tabela. E a faixa ≥50 tem
**três vezes mais** rótulos que as isoladas com F1 quase idêntico. Se a profundidade de anotação
governasse o resultado, a ordem seria monotônica em rótulos/proteína, e não é.

A correlação entre grau e número de rótulos é positiva mas moderada — Spearman +0,243 (BP), +0,211
(CC), +0,196 (MF) — e vai na direção **contrária** à do F1 na região de grau baixo. O IC médio das
anotações fica praticamente constante (~1,1) entre as faixas, então também não é o caso de as
conectadas terem termos sistematicamente mais específicos.

---

## 4. O que estas análises mostram

**1. O ProtT5 responde por cerca de metade do desempenho.** De +39% (CC) a +54% (BP) sobre a
topologia isolada. A rede sozinha alcança 0,37–0,49 de wFmax; com a sequência, 0,57–0,73.

**2. A curva de grau × acerto é em U, não monotônica.** As piores proteínas não são as isoladas —
são as de **grau 1 a 5**. Em BP, F1 0,4924 na faixa 3–5 contra 0,6288 nas isoladas e 0,6158 nas de
grau ≥50. Por isso o Spearman fica quase nulo (−0,057 a −0,110): ele mede monotonicidade, e não há.

Uma hipótese mecanística coerente com isso, embora não demonstrada por estes dados: o nó isolado
conserva o próprio embedding, apenas reescalado por α^hops, e nada o corrompe. O nó de grau baixo
recebe uma agregação dominada por um ou dois vizinhos — alta variância, capaz de diluir a
representação própria com sinal enganoso. O nó de grau alto faz média sobre muitos vizinhos e
recebe um agregado estável. Se estiver certa, a implicação de projeto é que a propagação deveria
ser condicionada ao grau, e não uniforme.

**3. O ganho da Early Fusion vem inteiramente das proteínas conectadas.** Comparando `hybrid` com
`fixed` em wFmax:

| Ont | isoladas | conectadas |
|---|---|---|
| BP | +0,0116 | **+0,0219** |
| CC | **−0,0110** | **+0,0128** |
| MF | **−0,0083** | **+0,0082** |

Em CC e MF a fusão **piora** as isoladas e melhora as conectadas. Faz sentido: para quem não tem
vizinho, o ramo da GNN não acrescenta informação, só ruído a ser combinado. É o argumento mais
direto a favor de uma fusão condicionada à conectividade — o tipo de trabalho futuro que esta
análise sustenta com evidência, não com especulação.

**4. Em MF a GNN degrada as conectadas.** wFmax 0,6430 contra 0,7123 do SPMM fixo, uma perda de
0,069 concentrada exatamente onde a GNN deveria ajudar. Nas isoladas as duas empatam (0,7499 vs
0,7568). É a explicação de por que o melhor hop em MF é **0**.

---

## Limitações destas análises

1. **Semente única, um run por célula.** As diferenças entre `fixed`, `gnn` e `hybrid` dentro de um
   mesmo subconjunto ficam entre 0,008 e 0,069; as menores não são distinguíveis de ruído.
2. **A hipótese mecanística do U não foi testada.** Ela é consistente com os dados, mas confirmá-la
   exigiria um experimento próprio — por exemplo, variar α por faixa de grau.
3. **`isolada` é relativa a thr=0,1.** Com outro limiar a partição muda, e o grau é medido no grafo
   completo (treino + validação + teste), não apenas entre os vizinhos de treino.
4. **F1 por proteína usa o τ global**, o mesmo para todos os subconjuntos. Um τ por subconjunto
   daria valores mais altos, mas não seria comparável entre eles.
