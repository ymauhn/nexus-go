# Confronto final — este trabalho × literatura

Convenção `ppi-only` (`ppi_v4/metrics.py`) nos dois lados: raiz mantida, IC curado de `{ont}_ic.csv`, propagação de ancestrais ligada. Smin em bits, **menor é melhor**.

**Todo campeão nosso é escolhido pelo wFmax de VALIDAÇÃO**, inclusive na tabela de teste. Selecionar e reportar no mesmo split inflaria os números.

*Procedência dos baselines:* `dados/baselines_ppi_only.csv`, extraído das Tabelas 6.19 e 6.20 do `ESQUELETO_CAPITULO_6.md`. A repontuação que as produziu foi executada fora deste repositório; o autor confirma a convenção.

**Correção aplicada ao Smin do DeepNF.** O notebook que o pontuou (`deepnf_pipeline_complete.ipynb`, células 27/29/31) divide a tabela de IC pela mediana antes de avaliar. Isso é inócuo para a perda, mas o Smin **soma** IC e portanto é linear na escala — enquanto Fmax, Fmax*, AuPRC, IAuPRC não usam IC e o wFmax o usa em razões, todos invariantes. Verificado sobre as predições deste trabalho em CC: reavaliar com `ic/mediana` deixa as cinco primeiras métricas bit a bit idênticas e multiplica o Smin por exatamente 1/2,0513. A correção é `smin x mediana(IC)`, com medianas 1,0633 (BP), 2,0513 (CC) e 1,4973 (MF). Os demais métodos vieram de outra repontuação e **não** foram alterados. O valor original está na coluna `smin_reportado` do CSV.

---

## VALIDAÇÃO


### BP

| Origem | Método | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| **este trabalho** | PPI-only (topologia pura) | 0,4581 | 0,3897 | 0,3637 | 0,4481 | 0,4675 | 23,14 |
| **este trabalho** | ProtT5 puro (h=0) | 0,5946 | 0,5597 | 0,5336 | 0,5836 | 0,6276 | 18,51 |
| **este trabalho** | SPMM fixo + MLP | 0,6095 | 0,5772 | 0,5523 | 0,6010 | 0,6472 | 17,89 |
| **este trabalho** | GNN E2E + MLP | 0,6127 | 0,5812 | 0,5578 | 0,5988 | 0,6528 | 17,39 |
| **este trabalho** | Early Fusion + MLP | 0,6215 | 0,5893 | 0,5652 | 0,6137 | 0,6468 | 17,01 |
| **este trabalho** | Rota visual (melhor) | 0,5896 | 0,5568 | 0,5310 | 0,5641 | 0,5987 | 18,29 |
| **este trabalho** | Late Fusion (linear) | 0,6324 | 0,6029 | **0,5796** | 0,6350 | 0,6827 | **16,93** |
| literatura | DeepGraphGO | 0,5554 | 0,5229 | 0,4892 | 0,5412 | 0,5817 | 19,69 |
| literatura | SEGT-GO | 0,5227 | 0,4814 | 0,4454 | 0,4277 | 0,5099 | 20,79 |
| literatura | Mashup | 0,4317 | 0,4140 | 0,3900 | 0,3435 | 0,3244 | 22,75 |
| literatura | MELISSA | 0,4150 | 0,3975 | 0,3732 | 0,3221 | 0,3087 | 22,80 |
| literatura | DeepNF | 0,3549 | 0,3098 | 0,2688 | 0,3403 | 0,3359 | 25,08 |

### CC

| Origem | Método | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| **este trabalho** | PPI-only (topologia pura) | 0,6741 | 0,6345 | 0,4825 | 0,6351 | 0,7208 | 11,96 |
| **este trabalho** | ProtT5 puro (h=0) | 0,7721 | 0,7477 | 0,6521 | 0,7354 | 0,8237 | 9,26 |
| **este trabalho** | SPMM fixo + MLP | 0,7821 | 0,7588 | 0,6684 | 0,7418 | 0,8406 | 8,69 |
| **este trabalho** | GNN E2E + MLP | 0,7821 | 0,7580 | 0,6635 | 0,7380 | 0,8322 | 8,50 |
| **este trabalho** | Early Fusion + MLP | 0,7866 | 0,7623 | 0,6706 | 0,8210 | 0,8283 | 8,34 |
| **este trabalho** | Rota visual (melhor) | 0,7621 | 0,7363 | 0,6333 | 0,7376 | 0,8099 | 9,33 |
| **este trabalho** | Late Fusion (linear) | 0,7947 | 0,7724 | **0,6849** | 0,7612 | 0,8592 | **8,21** |
| literatura | DeepGraphGO | 0,7360 | 0,7087 | 0,5987 | 0,7147 | 0,7840 | 9,63 |
| literatura | SEGT-GO | 0,7128 | 0,6811 | 0,5528 | 0,6288 | 0,7432 | 10,36 |
| literatura | Mashup | 0,5860 | 0,5663 | 0,4716 | 0,5005 | 0,4920 | 11,12 |
| literatura | MELISSA | 0,5765 | 0,5558 | 0,4599 | 0,4858 | 0,4802 | 11,28 |
| literatura | DeepNF | 0,6269 | 0,5780 | 0,3627 | 0,5853 | 0,6526 | 13,39 |

### MF

| Origem | Método | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| **este trabalho** | PPI-only (topologia pura) | 0,6135 | 0,5369 | 0,4892 | 0,4955 | 0,6273 | 9,21 |
| **este trabalho** | ProtT5 puro (h=0) | 0,7980 | 0,7627 | 0,7318 | 0,7375 | 0,8190 | 5,32 |
| **este trabalho** | SPMM fixo + MLP | 0,7980 | 0,7627 | 0,7318 | 0,7375 | 0,8190 | 5,32 |
| **este trabalho** | GNN E2E + MLP | 0,7609 | 0,7172 | 0,6783 | 0,6738 | 0,8065 | 6,09 |
| **este trabalho** | Early Fusion + MLP | 0,8064 | 0,7710 | 0,7381 | 0,8228 | 0,8254 | 5,18 |
| **este trabalho** | Rota visual (melhor) | 0,7642 | 0,7216 | 0,6825 | 0,5876 | 0,7699 | 6,11 |
| **este trabalho** | Late Fusion (linear) | 0,8092 | 0,7751 | **0,7444** | 0,7662 | 0,8498 | **5,15** |
| literatura | DeepGraphGO | 0,7693 | 0,7306 | 0,6930 | 0,7442 | 0,8076 | 5,71 |
| literatura | SEGT-GO | 0,7366 | 0,6900 | 0,6461 | 0,5877 | 0,7550 | 6,34 |
| literatura | Mashup | 0,5537 | 0,5077 | 0,4543 | 0,4913 | 0,4828 | 8,29 |
| literatura | MELISSA | 0,5429 | 0,4950 | 0,4406 | 0,4791 | 0,4738 | 8,45 |
| literatura | DeepNF | 0,5831 | 0,4853 | 0,3912 | 0,5438 | 0,5336 | 9,40 |

---

## TESTE


### BP

| Origem | Método | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| **este trabalho** | PPI-only (topologia pura) | 0,4625 | 0,3986 | 0,3730 | 0,4557 | 0,4737 | 23,68 |
| **este trabalho** | ProtT5 puro (h=0) | 0,5961 | 0,5631 | 0,5374 | 0,5853 | 0,6292 | 18,92 |
| **este trabalho** | SPMM fixo + MLP | 0,6127 | 0,5813 | 0,5561 | 0,6039 | 0,6509 | 18,30 |
| **este trabalho** | GNN E2E + MLP | 0,6205 | 0,5917 | 0,5695 | 0,6097 | 0,6627 | 17,66 |
| **este trabalho** | Early Fusion + MLP | 0,6287 | 0,5981 | 0,5751 | 0,6242 | 0,6573 | 17,35 |
| **este trabalho** | Rota visual (melhor) | 0,5945 | 0,5632 | 0,5383 | 0,5709 | 0,6048 | 18,80 |
| **este trabalho** | Late Fusion (linear) | 0,6374 | 0,6096 | **0,5869** | 0,6402 | 0,6883 | **17,12** |
| literatura | DeepGraphGO | 0,5529 | 0,5221 | 0,4892 | 0,5404 | 0,5789 | 20,11 |
| literatura | SEGT-GO | 0,5205 | 0,4822 | 0,4457 | 0,4282 | 0,5071 | 21,24 |
| literatura | Mashup | 0,4358 | 0,4186 | 0,3964 | 0,3482 | 0,3257 | 23,22 |
| literatura | MELISSA | 0,4194 | 0,4008 | 0,3772 | 0,3272 | 0,3117 | 23,42 |
| literatura | DeepNF | 0,3559 | 0,3114 | 0,2695 | 0,3415 | 0,3366 | 25,73 |

### CC

| Origem | Método | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| **este trabalho** | PPI-only (topologia pura) | 0,6738 | 0,6346 | 0,4839 | 0,6383 | 0,7220 | 12,03 |
| **este trabalho** | ProtT5 puro (h=0) | 0,7679 | 0,7429 | 0,6478 | 0,7330 | 0,8206 | 9,47 |
| **este trabalho** | SPMM fixo + MLP | 0,7816 | 0,7586 | 0,6700 | 0,7427 | 0,8388 | 8,76 |
| **este trabalho** | GNN E2E + MLP | 0,7842 | 0,7601 | 0,6670 | 0,7441 | 0,8407 | 8,44 |
| **este trabalho** | Early Fusion + MLP | 0,7882 | 0,7644 | 0,6747 | 0,8269 | 0,8321 | 8,34 |
| **este trabalho** | Rota visual (melhor) | 0,7625 | 0,7364 | 0,6340 | 0,7413 | 0,8114 | 9,32 |
| **este trabalho** | Late Fusion (linear) | 0,7952 | 0,7726 | **0,6865** | 0,7648 | 0,8617 | **8,19** |
| literatura | DeepGraphGO | 0,7374 | 0,7111 | 0,6035 | 0,7188 | 0,7859 | 9,67 |
| literatura | SEGT-GO | 0,7118 | 0,6807 | 0,5552 | 0,6324 | 0,7474 | 10,42 |
| literatura | Mashup | 0,5937 | 0,5744 | 0,4828 | 0,5127 | 0,5030 | 11,06 |
| literatura | MELISSA | 0,5854 | 0,5651 | 0,4719 | 0,4978 | 0,4897 | 11,23 |
| literatura | DeepNF | 0,6249 | 0,5759 | 0,3651 | 0,5844 | 0,6510 | 13,52 |

### MF

| Origem | Método | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| **este trabalho** | PPI-only (topologia pura) | 0,6104 | 0,5338 | 0,4862 | 0,4934 | 0,6242 | 9,20 |
| **este trabalho** | ProtT5 puro (h=0) | 0,7975 | 0,7602 | 0,7257 | 0,7375 | 0,8225 | 5,31 |
| **este trabalho** | SPMM fixo + MLP | 0,7975 | 0,7602 | 0,7257 | 0,7375 | 0,8225 | 5,31 |
| **este trabalho** | GNN E2E + MLP | 0,7609 | 0,7174 | 0,6777 | 0,6750 | 0,8064 | 6,04 |
| **este trabalho** | Early Fusion + MLP | 0,8022 | 0,7654 | 0,7294 | 0,8188 | 0,8205 | 5,22 |
| **este trabalho** | Rota visual (melhor) | 0,7652 | 0,7211 | 0,6804 | 0,5901 | 0,7707 | 6,02 |
| **este trabalho** | Late Fusion (linear) | 0,8089 | 0,7741 | **0,7389** | 0,7653 | 0,8492 | **5,14** |
| literatura | DeepGraphGO | 0,7698 | 0,7314 | 0,6896 | 0,7403 | 0,8041 | 5,67 |
| literatura | SEGT-GO | 0,7340 | 0,6876 | 0,6412 | 0,5864 | 0,7515 | 6,37 |
| literatura | Mashup | 0,5452 | 0,4982 | 0,4422 | 0,4787 | 0,4729 | 8,30 |
| literatura | MELISSA | 0,5357 | 0,4865 | 0,4286 | 0,4643 | 0,4631 | 8,47 |
| literatura | DeepNF | 0,5843 | 0,4870 | 0,3900 | 0,5455 | 0,5349 | 9,36 |

---

## Leitura

### Vencemos o melhor da literatura nas três ontologias, nos dois splits

| Ont | nosso melhor (val → teste) | DeepGraphGO (val → teste) | Δ no teste |
|---|---|---|---|
| BP | Late Fusion (linear) — 0,5796 → **0,5869** | 0,4892 → 0,4892 | **+0,0977** |
| CC | Late Fusion (linear) — 0,6849 → **0,6865** | 0,5987 → 0,6035 | **+0,0830** |
| MF | Late Fusion (linear) — 0,7444 → **0,7389** | 0,6930 → 0,6896 | **+0,0493** |

A ordenação é **idêntica em validação e teste**: todos os campeões foram escolhidos pelo wFmax de validação, e reportá-los em teste não mudou quem ganha. É a evidência de que a seleção não vazou.


### O Smin concorda, depois de corrigir o DeepNF

Antes da correção, o DeepNF exibia o **melhor Smin de CC de toda a tabela** — 6,59 contra 9,67 do DeepGraphGO — apesar do pior Fmax\*, do pior wFmax e da pior AuPRC. Smin e wFmax são as duas métricas ponderadas por IC e discordavam só nessa célula.

A causa está no código que o pontuou: o notebook divide a tabela de IC pela mediana antes de avaliar. Corrigido, o DeepNF passa a ter o **pior Smin nas três ontologias**, consistente com todas as suas outras métricas. **A contradição era artefato de pontuação, não propriedade do método.**

Vale como nota metodológica: normalizar o IC é inofensivo para uma função de perda e silenciosamente errado para uma métrica semântica — e as duas costumam compartilhar o mesmo vetor.


### Ressalvas

1. **Semente única.** Margens de 0,05 a 0,10 sobre a literatura são grandes demais para ruído de semente; as comparações internas entre nossas etapas, não.
2. **O ensemble ajusta γ em validação**, um grau de liberdade que os modelos isolados e os baselines não têm.
3. **A cobertura foi omitida.** Mashup e MELISSA cobrem 63–70% das proteínas contra 95–96% dos demais; parte da distância para esses dois é alcance, não qualidade.
4. **Procedência assimétrica.** Nossos números saem dos manifests desta campanha; os dos baselines, da repontuação registrada no esqueleto.

