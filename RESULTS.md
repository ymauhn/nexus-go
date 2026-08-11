# Resultados do Capítulo 6 — índice

Tudo o que a pesquisa produziu, consolidado. Gerado em 2026-08-05 a partir dos artefatos das
execuções, não transcrito à mão.

| Arquivo | Conteúdo | Runs |
|---|---|---|
| `01_TABELAS_dedupmax.md` | Campanha principal: ProtT5 + PPI, regime `dedup=max` | 96 |
| `02_PPI_ONLY_TABELAS.md` | PPI-only: ablações A, B e C, **validação e teste** | 33 |
| `02_PPI_ONLY.md` | PPI-only: texto analítico e custo | 33 |
| `03_TABELAS_dedupsum.md` | Referência histórica: regime `dedup=sum` | 6 + ensemble |
| `04_APENDICE_AMBIENTE.md` | Hardware, versões de bibliotecas, hiperparâmetros | — |
| `05_ANALISES_EXTRAS.md` | Ganho do ProtT5, isoladas vs conectadas | 9 rotas |
| `06_GRAU_teste.md` | Conectividade e grau × acerto, **8 grupos** | 24 rotas |
| `07_CONFRONTO_FINAL.md` | **Nossos campeões × literatura**, validação e teste | 7 + 5 |
| `08_PROTOCOLO.md` | Escala dos dados, rede PPI, config; literatura na conv. `deepgraphgo` | — |
| `dados/` | Os CSVs de origem, para reprodução | — |

**Regenerar** (de `experimentos_finais_dissertacao/pasta_mais_importante/`):

```bash
python scripts_v4/tabelas_cap6.py --csv campanha_dedupmax/tabelas/results_all.csv --results-dir campanha_dedupmax/results --ppi-only ../../projeto_ppi_only/results/ABLATIONS_summary.csv --out ../../RESULTADOS_CAPITULO_6/01_TABELAS_dedupmax.md
```

Célula `—` significa run ausente. Célula `n/a` significa não aplicável por construção. As duas coisas
são distintas e o gerador nunca inventa um valor para nenhuma delas.

---

## O que cada tabela compara

### 01 — Campanha `dedup=max` (96 runs, 121,5 GPU-horas)

| Tabela | O eixo comparado | Células | Serve para |
|---|---|---|---|
| **6.8** | hops de 0 a 10, SPMM fixo + MLP, α=0,5 | 33 | Localizar o melhor número de saltos e mostrar que o ganho satura cedo |
| **6.10** | α ∈ {0,1 … 0,9} no melhor hop | 15 | Quanto da representação própria do nó preservar |
| **6.11** | GCN vs GAT fim-a-fim + MLP, val e teste, com custo | 12 | A GNN treinada vale o preço? |
| **6.11b** | GCN vs GAT nas 4 combinações método × cabeça | 12 | A preferência de arquitetura depende do contexto? |
| **6.12** | Early Fusion: `add`/`concat` × GCN/GAT | 12 | Fundir SPMM com GNN ajuda? |
| **6.12b** | `add` vs `concat` isolado | 6 | A operação de fusão importa? |
| **6.13** | Late Fusion: 11 estratégias de ensemble | 33 | Combinar predições supera combinar representações? |
| **6.14** | hops com cabeça visual (grade grossa) | 21 | O ótimo de hops se desloca ao trocar a cabeça? |
| **6.15** | α, método e GNN na rota visual | 25 | Panorama completo da rota visual |
| **6.16** | Custo: horas, VRAM, épocas por família | 6 | O argumento de eficiência |
| **síntese** | Melhor de cada rota, escolhido em val, reportado em teste | 18 | A tabela que resume o capítulo |
| **verificação** | Diferença teste − validação nas 96 runs | — | Não há vazamento para o split de teste |

### 02 — PPI-only (33 runs)

| Ablação | O eixo comparado | Células |
|---|---|---|
| **A** | BCE vs ProteinLoss (MLP, pooling `max`) | 6 |
| **B** | pooling {max, avg, raw} × {MLP, CNN1D} | 18 |
| **C** | ConvNeXt-Tiny vs ResNet50 × {max, avg} | 12 |

### 03 — Regime `dedup=sum` (histórico)

Seis modelos-base mais o ensemble completo, do Exame de Qualificação e do VISAPP. Serve para
mostrar que a mudança de regra de deduplicação não inverteu nenhuma conclusão. As demais tabelas
saem vazias porque as famílias `hybrid` e visual não foram executadas nesse regime — é ausência de
experimento, não de dado.

---

## As cinco conclusões que os dados sustentam

**1. A GCN vence a GAT em 12 de 12 comparações.** Sem uma única exceção, atravessando GNN
fim-a-fim e Early Fusion, cabeça vetorial e visual, as três ontologias. As margens vão de +0,0139 a
+0,0931 de wFmax. A GAT ainda custa 55% mais tempo e 46% mais memória. Para um mecanismo de atenção
que deveria aprender pesos melhores que os do STRING, é um resultado que merece discussão.

**2. A propagação fixa empata com a GNN treinada por 1/19 do custo.** SPMM fixo entrega 0,32 h
medianas por run contra 6,09 h da GNN, com 6,6 GB contra 21,2 GB de VRAM — e o wFmax fica dentro de
0,01 em BP e à frente em CC e MF. É a tese central do trabalho, e a Tabela 6.16 é a evidência.

**3. A rota vetorial vence a visual nas três ontologias, em todos os métodos.** Sem inversão em
nenhuma das 18 células da síntese. Somado ao resultado do PPI-only, onde as cabeças de imagem
também foram as piores, a conclusão é consistente entre duas gerações independentes de
experimentos.

**4. A Late Fusion linear supera todas as alternativas elaboradas.** Uma média ponderada por um
único escalar γ ≈ 0,5 bate as estratégias por IC, por grau, a mestra e cinco variantes de stacking,
nas três ontologias. As importâncias do stacking explicam: `IC_j` e `log(D+1)` recebem peso ~0,00.
As meta-features em que as estratégias 3, 4 e 5 se apoiam não carregam informação útil aqui.

**5. Em MF o melhor hop é 0.** A propagação não ajuda nessa ontologia — o ProtT5 sozinho já carrega
o sinal. O arquivo 05 mostra por quê: a GNN perde 0,069 de wFmax justamente nas proteínas
**conectadas**, onde deveria ajudar.

**6. A curva de grau × acerto é em U.** As proteínas pior preditas não são as isoladas, e sim as de
grau 1 a 5. O ganho da Early Fusion vem inteiramente das conectadas — em CC e MF ela chega a
**piorar** as isoladas. Detalhes no arquivo 05.

---

## Limitações a declarar

1. **Semente única (1337), um run por célula.** Sem intervalo de confiança. Diferenças abaixo de
   ~0,005 de wFmax não são interpretáveis — o que inclui `add` vs `concat` (Tabela 6.12b) e boa
   parte da Ablação C do PPI-only.
2. **O ensemble ajusta γ em validação**, um parâmetro a mais que os modelos isolados não têm. Com
   ganhos de 0,013 a 0,022 isso não anula o resultado, mas precisa ser dito.
3. **Hardware heterogêneo** — cinco configurações, detalhadas no arquivo 04. Métricas comparáveis
   entre todas; **tempos não**, e a Tabela 6.16 mistura RTX A5500 com GTX 1080 Ti.
4. **`concat` só foi executado na cabeça vetorial**, por decisão de projeto. A rota visual tem
   apenas `add`.
5. **Duas convenções de métrica** convivem no capítulo. As tabelas aqui usam a convenção
   `ppi_v4/metrics.py` — raiz mantida, IC curado, propagação de ancestrais. O confronto com a
   literatura exige repontuar os baselines na mesma convenção.

---

## O que ainda falta

| Item | Estado |
|---|---|
| Tabelas 6.1–6.3 (protocolo) | **Prontas** — arquivo 08, calculadas dos dados |
| Tabelas 6.4–6.7 (PPI-only) | **Prontas** — arquivo 02, val e teste |
| Tabelas 6.8–6.16 (campanha) | **Prontas** — arquivo 01 |
| Tabelas 6.17–6.18 (viés de cobertura) | **Prontas** — arquivo 08 |
| Tabelas 6.19–6.21 (confronto final) | **Prontas** — arquivo 07 |
| Tabela 6.9 (controle negativo `sum` vs `max`) | só `mf/h0` existe nos dois regimes |
| MELISSA em MF na convenção `deepgraphgo` | `pending` no `tabela_6metricas.json`; não afeta o arquivo 07 |
| Versões exatas de PyTorch/CUDA do servidor | Comando no arquivo 04, §A.2.2 |
| Nome e VRAM da GPU do Colab | Comando no arquivo 04, §A.1.1 |

**Regenerar as análises do arquivo 05** (precisam das probs em `campanha_dedupmax/results/`):

```bash
python scripts_v4/analises_extras.py --results-dir campanha_dedupmax/results --data-dir dados/dados --csv campanha_dedupmax/tabelas/results_all.csv --ppi-only ../../projeto_ppi_only/results/ABLATIONS_summary.csv --out ../../RESULTADOS_CAPITULO_6/05_ANALISES_EXTRAS.md
```
