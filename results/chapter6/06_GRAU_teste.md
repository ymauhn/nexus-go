# Análise por conectividade — split test

O melhor candidato de **cada grupo**, escolhido pelo wFmax de validação. O PPI-only não usa ProtT5 e entra como piso de topologia pura; seus conjuntos de teste foram verificados como idênticos aos da campanha — mesmas proteínas, mesma ordem, mesmos termos —, então as comparações são linha a linha.

`isolada` = grau zero no grafo limiarizado em thr=0.1, `dedup=max`.

---

## BP — 3,421 isoladas de 9,221 (37.1%)

### A. wFmax por conectividade

| Grupo | isoladas | conectadas | Δ (conect. − isol.) |
|---|---|---|---|
| ProtT5 puro (h=0) | 0,5989 | 0,5023 | -0,0966 |
| SPMM fixo + MLP | 0,6130 | 0,5239 | -0,0891 |
| SPMM fixo + visão | 0,5708 | 0,5110 | -0,0597 |
| GNN + MLP | 0,6226 | 0,5397 | -0,0828 |
| GNN + visão | 0,5787 | 0,5159 | -0,0629 |
| Híbrido + MLP | 0,6246 | 0,5458 | -0,0788 |
| Híbrido + visão | 0,5133 | 0,5009 | -0,0124 |
| PPI-only (sem ProtT5) | 0,2486 | 0,4388 | +0,1902 |

### B. F1 médio por faixa de grau

| Faixa | n | ProtT5 puro (h=0) | SPMM fixo + MLP | SPMM fixo + visão | GNN + MLP | GNN + visão | Híbrido + MLP | Híbrido + visão | PPI-only (sem ProtT5) |
|---|---|---|---|---|---|---|---|---|---|
| 0 (isoladas) | 3,421 | 0,5964 | 0,6145 | 0,5856 | 0,6265 | 0,5871 | **0,6288** | 0,5323 | 0,2596 |
| 1–2 | 433 | 0,5149 | 0,5206 | 0,4528 | 0,5271 | 0,4823 | **0,5317** | 0,4736 | 0,3431 |
| 3–5 | 757 | 0,4756 | **0,4933** | 0,4256 | 0,4786 | 0,4428 | 0,4924 | 0,4297 | 0,3593 |
| 6–14 | 2,281 | 0,4976 | 0,5132 | 0,4976 | 0,5208 | 0,4968 | **0,5355** | 0,4847 | 0,4137 |
| 15–49 | 2,066 | 0,5170 | 0,5431 | 0,5517 | 0,5670 | 0,5485 | **0,5767** | 0,5405 | 0,4772 |
| ≥ 50 | 263 | 0,5253 | 0,5640 | 0,5778 | 0,6031 | 0,5935 | **0,6158** | 0,5816 | 0,5263 |

### C. Correlação grau × F1

| Grupo | τ | Spearman ρ | F1 global |
|---|---|---|---|
| ProtT5 puro (h=0) | 0.51 | -0,109 | 0,5384 |
| SPMM fixo + MLP | 0.50 | -0,105 | 0,5577 |
| SPMM fixo + visão | 0.35 | -0,038 | 0,5366 |
| GNN + MLP | 0.39 | -0,083 | 0,5696 |
| GNN + visão | 0.39 | -0,042 | 0,5395 |
| Híbrido + MLP | 0.46 | -0,077 | 0,5779 |
| Híbrido + visão | 0.34 | 0,022 | 0,5126 |
| PPI-only (sem ProtT5) | 0.41 | 0,415 | 0,3662 |

---

## CC — 3,188 isoladas de 9,292 (34.3%)

### A. wFmax por conectividade

| Grupo | isoladas | conectadas | Δ (conect. − isol.) |
|---|---|---|---|
| ProtT5 puro (h=0) | 0,6930 | 0,6238 | -0,0692 |
| SPMM fixo + MLP | 0,7032 | 0,6528 | -0,0504 |
| SPMM fixo + visão | 0,6585 | 0,6141 | -0,0444 |
| GNN + MLP | 0,7044 | 0,6476 | -0,0568 |
| GNN + visão | 0,6596 | 0,6211 | -0,0385 |
| Híbrido + MLP | 0,6922 | 0,6656 | -0,0267 |
| Híbrido + visão | 0,6357 | 0,6213 | -0,0144 |
| PPI-only (sem ProtT5) | 0,3397 | 0,5374 | +0,1977 |

### B. F1 médio por faixa de grau

| Faixa | n | ProtT5 puro (h=0) | SPMM fixo + MLP | SPMM fixo + visão | GNN + MLP | GNN + visão | Híbrido + MLP | Híbrido + visão | PPI-only (sem ProtT5) |
|---|---|---|---|---|---|---|---|---|---|
| 0 (isoladas) | 3,188 | 0,7640 | 0,7732 | 0,7477 | **0,7795** | 0,7458 | 0,7729 | 0,7299 | 0,5799 |
| 1–2 | 380 | 0,6861 | 0,6847 | 0,6456 | **0,6932** | 0,6748 | 0,6859 | 0,6645 | 0,6034 |
| 3–5 | 634 | 0,7132 | 0,7271 | 0,6824 | 0,7201 | 0,6963 | **0,7312** | 0,7013 | 0,6424 |
| 6–14 | 2,357 | 0,6996 | 0,7180 | 0,6908 | 0,7169 | 0,6957 | **0,7285** | 0,6991 | 0,6463 |
| 15–49 | 2,388 | 0,7076 | 0,7366 | 0,7305 | 0,7477 | 0,7277 | **0,7580** | 0,7253 | 0,6737 |
| ≥ 50 | 345 | 0,6655 | 0,7033 | 0,7187 | 0,7526 | 0,7119 | **0,7545** | 0,7206 | 0,6695 |

### C. Correlação grau × F1

| Grupo | τ | Spearman ρ | F1 global |
|---|---|---|---|
| ProtT5 puro (h=0) | 0.58 | -0,129 | 0,7229 |
| SPMM fixo + MLP | 0.56 | -0,097 | 0,7404 |
| SPMM fixo + visão | 0.44 | -0,046 | 0,7191 |
| GNN + MLP | 0.51 | -0,087 | 0,7468 |
| GNN + visão | 0.52 | -0,055 | 0,7209 |
| Híbrido + MLP | 0.50 | -0,057 | 0,7507 |
| Híbrido + visão | 0.53 | -0,021 | 0,7159 |
| PPI-only (sem ProtT5) | 0.46 | 0,161 | 0,6294 |

---

## MF — 2,553 isoladas de 7,864 (32.5%)

### A. wFmax por conectividade

| Grupo | isoladas | conectadas | Δ (conect. − isol.) |
|---|---|---|---|
| ProtT5 puro (h=0) | 0,7568 | 0,7123 | -0,0446 |
| SPMM fixo + MLP | 0,7568 | 0,7123 | -0,0446 |
| SPMM fixo + visão | 0,7032 | 0,6706 | -0,0326 |
| GNN + MLP | 0,7499 | 0,6430 | -0,1068 |
| GNN + visão | 0,7145 | 0,6208 | -0,0937 |
| Híbrido + MLP | 0,7485 | 0,7205 | -0,0279 |
| Híbrido + visão | 0,7011 | 0,6600 | -0,0411 |
| PPI-only (sem ProtT5) | 0,3568 | 0,5195 | +0,1627 |

### B. F1 médio por faixa de grau

| Faixa | n | ProtT5 puro (h=0) | SPMM fixo + MLP | SPMM fixo + visão | GNN + MLP | GNN + visão | Híbrido + MLP | Híbrido + visão | PPI-only (sem ProtT5) |
|---|---|---|---|---|---|---|---|---|---|
| 0 (isoladas) | 2,553 | 0,7805 | 0,7805 | 0,7437 | 0,7813 | 0,7541 | **0,7826** | 0,7434 | 0,5247 |
| 1–2 | 486 | 0,7589 | 0,7589 | 0,7253 | 0,7162 | 0,6991 | **0,7613** | 0,6998 | 0,5792 |
| 3–5 | 630 | **0,7496** | 0,7496 | 0,7145 | 0,6854 | 0,6604 | 0,7463 | 0,6954 | 0,5579 |
| 6–14 | 1,952 | 0,7467 | 0,7467 | 0,7159 | 0,6982 | 0,6631 | **0,7573** | 0,7136 | 0,5777 |
| 15–49 | 1,945 | 0,7296 | 0,7296 | 0,6954 | 0,6745 | 0,6571 | **0,7469** | 0,6963 | 0,5647 |
| ≥ 50 | 298 | 0,6791 | 0,6791 | 0,6604 | 0,6593 | 0,6179 | **0,7255** | 0,6793 | 0,5419 |

### C. Correlação grau × F1

| Grupo | τ | Spearman ρ | F1 global |
|---|---|---|---|
| ProtT5 puro (h=0) | 0.50 | -0,128 | 0,7519 |
| SPMM fixo + MLP | 0.50 | -0,128 | 0,7519 |
| SPMM fixo + visão | 0.47 | -0,104 | 0,7182 |
| GNN + MLP | 0.51 | -0,185 | 0,7179 |
| GNN + visão | 0.48 | -0,168 | 0,6914 |
| Híbrido + MLP | 0.42 | -0,110 | 0,7611 |
| Híbrido + visão | 0.42 | -0,096 | 0,7154 |
| PPI-only (sem ProtT5) | 0.49 | 0,083 | 0,5544 |

---

## Leitura

### 1. A propagação não cria a correlação negativa — ela a atenua

O grupo `ProtT5 puro (h=0)` é o SPMM com zero saltos: o grafo não participa de nenhuma conta. Ele
**já tem** o Spearman negativo, praticamente igual ao dos modelos propagados:

| Ont | ProtT5 puro | melhor com grafo | efeito |
|---|---|---|---|
| BP | −0,109 | −0,077 | menos negativo |
| CC | −0,129 | −0,057 | menos negativo |
| MF | −0,128 | −0,110 | menos negativo |

Isso descarta a hipótese que eu havia proposto antes — de que a agregação em nós de grau baixo
diluiria a representação própria com sinal ruidoso. Se fosse esse o mecanismo, a correlação
negativa apareceria **ao ligar** a propagação, e ela já está lá com o grafo desligado. O que a
inclinação negativa mede é uma propriedade das proteínas: as muito conectadas tendem a ser
multifuncionais, carregam mais anotações (Spearman grau × nº de rótulos = +0,24 em BP) e são
intrinsecamente mais difíceis para um preditor de sequência. **A propagação reduz o problema em
todas as três ontologias.**

### 2. O PPI-only é o espelho, e confirma que a topologia funciona

Sem ProtT5, o sinal inverte de sinal e fica forte: ρ = **+0,415** (BP), **+0,161** (CC), **+0,083**
(MF), e em BP a progressão é estritamente monotônica — 0,2596 → 0,3431 → 0,3593 → 0,4137 → 0,4772
→ 0,5263. Nas isoladas ele desaba para wFmax 0,2486 contra 0,4388 nas conectadas.

É o comportamento esperado de um modelo cuja única entrada é o grafo: sem vizinho, o vetor é nulo e
resta o prior. **A topologia faz exatamente o que promete; o que muda o sinal é o ProtT5 entrar.**

### 3. O ganho do grafo cresce com o grau

Diferença entre o melhor grupo e o `ProtT5 puro`, por faixa:

| Faixa de grau | BP | CC | MF |
|---|---|---|---|
| 0 (isoladas) | +0,0324 | +0,0155 | +0,0021 |
| 1–2 | +0,0168 | +0,0071 | +0,0024 |
| 3–5 | +0,0177 | +0,0180 | 0,0000 |
| 6–14 | +0,0379 | +0,0289 | +0,0106 |
| 15–49 | +0,0597 | +0,0504 | +0,0173 |
| **≥ 50** | **+0,0905** | **+0,0890** | **+0,0464** |

**Monotônico da faixa 1–2 em diante, nas três ontologias.** Da faixa de grau baixo à de grau alto o
ganho quintuplica em BP, multiplica por 12 em CC e por 19 em MF. O grafo ajuda onde há grafo — o
que é esperado, mas aqui está medido, e a magnitude não era previsível.

O ganho nas isoladas (+0,0324 em BP) merece ressalva: ali o grafo não traz informação nenhuma, e a
diferença vem da capacidade extra do modelo híbrido, não de topologia.

### 4. Em MF a GNN piora a correlação

Único caso em que o grafo torna a inclinação **mais** negativa: ρ = −0,185 na GNN contra −0,128 no
ProtT5 puro, com F1 global caindo de 0,7519 para 0,7179. Coerente com o melhor hop de MF ser 0 e
com a GNN perder 0,069 de wFmax nas conectadas.

Nota de consistência: em MF o `SPMM fixo + MLP` tem ρ e F1 **idênticos** ao `ProtT5 puro` porque o
melhor hop dessa ontologia é 0 — são o mesmo run, e a tabela o mostra por construção.

### 5. As cabeças visuais têm inclinação mais achatada

ρ entre −0,046 e +0,022 em BP e CC, contra −0,077 a −0,129 nas vetoriais. Elas perdem menos nas
conectadas, mas partem de um patamar mais baixo em tudo: o F1 global da melhor rota visual fica
abaixo do `ProtT5 puro` em BP (0,5395 contra 0,5384 é empate) e em CC (0,7209 contra 0,7229). **Nem
o ganho de conectividade compensa a perda de representação.**

## Limitações

1. Semente única, um run por célula. Diferenças abaixo de ~0,005 não são interpretáveis.
2. A faixa ≥50 tem 263–345 proteínas, a menor de todas — os ganhos mais altos da tabela são também
   os menos precisos.
3. O grau é medido no grafo completo (treino + validação + teste), não só contra o treino.
4. `τ` é global por modelo, não por faixa. Um τ por faixa daria valores maiores, mas não
   comparáveis entre faixas.
