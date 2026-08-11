# Instruções de Execução — Ablações PPI-only (código já pronto e testado)

> **Para o agente (claude-code):** o pacote `ppi_only/` **já foi escrito e validado** (plumbing
> testado ponta a ponta — ver §0). Sua tarefa **NÃO é reescrever** a pipeline, e sim: (1) posicionar
> os arquivos e verificar o ambiente, (2) rodar os *gates* de sanidade e mockup, (3) executar a
> matriz de ablações nos dados/GPU reais, (4) salvar os artefatos e gerar os relatórios. Corrija
> apenas problemas **de ambiente/execução** (versões de lib, download de pesos, caminhos). **Não
> altere** a metodologia (métrica, loss, base-de-treino do vetor, blocos removidos). Pare e reporte
> se um *gate* falhar.

---

## 0. O que já foi feito e validado (contexto)

Este pacote foi reconstruído por engenharia reversa do notebook legado
`PPI_Only_early_Experiments.ipynb` e **testado de ponta a ponta num dataset sintético** (CPU),
com **sucesso** em:

- `py_compile` de todos os módulos (sintaxe OK).
- **Sanity checks** (§6) passaram para MLP, CNN1D e ConvNeXt-Tiny.
- **Mockups** (1 época, subset): MLP, CNN1D e imagem (ConvNeXt-Tiny) → treino + val + test + salvamento OK.
- Caminho **`raw`** (input_dim = train_size) e **BCE** funcionam.
- **Determinismo confirmado**: duas execuções idênticas produziram métricas idênticas (a correção
  de *seeding* do DataLoader funciona).
- **Orquestrador** `run_ablations`: monta a matriz, deduplica (o run `mlp_max_protein` aparece
  como `A+B`), agrega em `ABLATIONS_summary.{csv,md}` e gera `RUN_REPORT.md`.

Portanto: o código está correto no nível de *plumbing*. **O que falta é rodar de verdade** (GPU,
dados reais, 200 épocas). É isso que você fará. Um detalhe de ambiente já tratado: `np.trapz` foi
removido no NumPy 2.x; `ppi_only/compat.py` já aplica um *alias* para `np.trapezoid` se necessário.

---

## 1. Objetivo e contexto do projeto

Dissertação de mestrado em anotação funcional de proteínas (GO: **BP**, **CC**, **MF**). A geração
**legada PPI-only** (Exame de Qualificação) foi refatorada num pacote limpo para produzir
*baselines* consistentes e ablações para o capítulo de Resultados. Fatos estabelecidos (assuma
como verdade — já embutidos no código):

- **Rede PPI** (`ppi.csv`): `combined_score` do **STRING**, filtrado em `>= 400` e dividido por
  1000 → scores em **[0.400, 0.999]**; **791.011 arestas únicas** após dedup por par (score máx)
  + remoção de self-loops. **Sem novo threshold de confiança** (o piso 0.4 já é o corte).
- **Feature PPI-only:** perfil de escores contra o **conjunto de treino apenas** (salvaguarda
  anti-leakage). Dimensão `N = #treino`. Self-loop 1.0 é **diagonal** (não vira `N+1`).
- **~33–38 %** das proteínas não têm vizinho de treino (linha nula) — esperado, não é bug.
- **Métrica e loss** são **reutilizadas** do pacote v32 (`ppi_v4`) — nunca reimplemente.

---

## 2. Ambiente (verifique; ative antes de rodar)

Terminal aberto na **raiz do projeto**, ambiente Python ativado. Verifique:

```bash
python -c "import torch, torchvision, numpy, pandas, sklearn, matplotlib, yaml; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

- Se faltar lib, instale via `requirements.txt`/`environment.yml`.
- **Pesos pré-treinados:** na 1ª execução dos heads de imagem, o torchvision baixa os pesos de
  ResNet50/ConvNeXt-Tiny (rede necessária). Se não houver rede, ou rode só os heads vetoriais, ou
  pré-baixe os pesos. Para *smoke* sem rede, use `--override pretrained=0`.
- Se `cuda=False` e não houver GPU: rodar em CPU é possível mas **muito lento** para 200 épocas —
  reporte ao usuário antes de rodar a matriz completa.

---

## 3. Estrutura de diretórios esperada

```
projeto/                              <-- terminal AQUI
├── instrucoes_experimentos.md        # este arquivo
├── ppi_only/                         # (FORNECIDO) pacote refatorado + testado
│   ├── __init__.py  compat.py  config.py  graph.py  image.py
│   ├── datasets.py  models.py  sanity.py  train.py  main.py  run_ablations.py
├── configs/ppi_only/                 # (FORNECIDO) base.yaml + mock_*.yaml
├── ppi_v4/ppi_v4/                    # (OBRIGATÓRIO) pacote v32: metrics/losses/engine/models/data
├── data/
│   ├── ppi.csv
│   ├── {bp,cc,mf}_{train,val,test}.csv   e   {bp,cc,mf}_ic.csv
│   └── go.obo                        # OBRIGATÓRIO (propagação GO nas métricas)
├── requirements.txt / environment.yml
└── (criados na execução) runs/  results/  figures/  logs/
```

Se `data/go.obo`, algum CSV, ou o pacote `ppi_v4` faltar, **pare e reporte** exatamente o que falta.
(O `ppi_only/compat.py` procura o `ppi_v4` na raiz do projeto automaticamente.)

---

## 4. Restrições arquiteturais (já implementadas — NÃO altere)

- **Removidos:** `equalize` (equalização 1D), `dilate`, `enhance` (imagem), `convexize` (grafo),
  e **todos** os canais ProtT5. Não reintroduza.
- **Base-de-treino anti-leakage:** `graph.build_relations_dict` indexa só o treino; vizinho só
  entra se for do treino; self-loop 1.0 diagonal (dim = `N`, não `N+1`). **Mantenha.**
- **`pooling`:** `max` / `avg` (→ vetor 1024) e **`raw`** = sem pooling (classificador consome o
  vetor completo de dimensão `N = train_size`; varia por domínio). `raw` só para MLP/CNN1D.
- **Loss/métrica:** `compat.py` reexporta `build_criterion`, `evaluate_collect` etc. do `ppi_v4`.
- **Reprodutibilidade:** `set_global_seed(1337)` + `DataLoader` com `generator` e `worker_init_fn`
  (já implementado em `datasets.build_loader`). Sem `weight_decay` (consistente com o legado).
- **Head de imagem:** o vetor é *poolado* para `image_size` (default **224**) e a imagem é a
  matriz outer-diff `|v_i − v_j|` (`image_size × image_size`), sem dilate/enhance. `image_size` é
  configurável; se mudar, **logue**.

---

## 5. Mapa dos módulos (para orientação, não para reescrever)

| arquivo | papel |
|---|---|
| `config.py` | defaults + merge YAML/CLI + `make_tag` + validação |
| `compat.py` | ponte para `ppi_v4` (métrica/loss/engine/backbones) + shim `np.trapz` |
| `graph.py` | `build_relations_dict`, `make_vector`, `pool_to_size`, `count_null_vectors` |
| `image.py` | `outer_diff_image` (grayscale puro), `image_tensor_from_vec` |
| `datasets.py` | `VectorDataset`, `ImageDataset`, `build_loader` (determinístico) |
| `models.py` | `MLP`, `CNN1D`, `build_model` (imagem via `ppi_v4`) |
| `sanity.py` | 6 checks obrigatórios (§6) |
| `train.py` | `run_experiment`: treina (via `engine.fit`) + avalia val/test + salva artefatos |
| `main.py` | CLI de 1 experimento (com gate de sanity) |
| `run_ablations.py` | matriz A/B/C + agregação + `RUN_REPORT.md` |

---

## 6. Sanity checks (gate) — automáticos

`ppi_only/sanity.py` roda **automaticamente** antes de treinar (em `main.py`). Ele checa:
alinhamento (vetor = `N`, sem val na base-treino), % de isoladas (~33–38 %), shapes/faixas de
batch, simetria e diagonal-nula da imagem, contagem de parâmetros, e loss finita para `protein` e
`bce`. Para rodar só o sanity: `python -m ppi_only.main --sanity-only --domain bp --model resnet50`.
**Se o sanity falhar, `main.py` aborta com código 2** — pare e reporte.

---

## 7. Gate de mockup (rodar ANTES da matriz completa)

```bash
python -m ppi_only.main --config configs/ppi_only/mock_mlp.yaml   --domain bp --smoke
python -m ppi_only.main --config configs/ppi_only/mock_cnn1d.yaml --domain bp --smoke
python -m ppi_only.main --config configs/ppi_only/mock_image.yaml --domain bp --arch convnext_tiny --smoke
# opcional sem rede: acrescente  --override pretrained=0
```

`--smoke` já força `max_epochs=1`, `limit_train=512`, `limit_val=256`. **Só prossiga se os 3
passarem.** (No sintético, os 3 passaram; nos dados reais, valida caminhos de dados e memória.)

---

## 8. Matriz de ablações (após o mockup)

**Comum:** `loss=protein` (salvo onde varia), `pos_weight_mode=ic`, Adam, `lr=1e-4`,
`batch_size=32`, `max_epochs=200`, `patience=25`, `out_size=1024`, `seed=1337`, sem weight_decay.
**Domínios:** `bp, cc, mf`. Avalia **val e test**.

- **A — MLP:** BCE vs ProteinLoss (pooling `max`).
- **B — MLP e CNN1D:** pooling `max` / `avg` / `raw` (loss `protein`).
- **C — ResNet50 e ConvNeXt-Tiny:** pooling `max` / `avg` (loss `protein`).

Rodar tudo de uma vez (deduplica automaticamente; ~34–36 runs):

```bash
python -m ppi_only.run_ablations --domains bp,cc,mf
```

Se GPU/tempo forem limitados, faça por partes e **logue as pendências** (nunca truncar em
silêncio):

```bash
python -m ppi_only.run_ablations --domains bp                    # BP completo primeiro
python -m ppi_only.run_ablations --domains bp --only-model mlp   # ou por modelo (mais barato 1º)
python -m ppi_only.run_ablations --domains cc,mf
```

Ordem de custo crescente já aplicada: `mlp → cnn1d → convnext_tiny → resnet50`.
**Nome do run:** `{domain}_{model}_{pooling}_{loss}_seed1337`.

---

## 9. Artefatos salvos (automático por run) — já validado

- **Pesos (melhor val):** `runs/{tag}_best.pt`.
- **Curva de loss treino×val:** `runs/{tag}_losses.png` + `runs/{tag}_loss_hist.json`.
- **6 métricas (val e test):** `results/{tag}_metrics.json` e `results/{tag}_metrics.csv`
  (colunas: `split, fmax, fmax_star, wfmax, smin, auprc, iauprc`).
- **Manifesto:** `results/{tag}_manifest.json` (config, seed, versões, `input_dim`, `train_size`,
  `null_train_vectors`, tempo).
- **Agregados (pelo `run_ablations`):** `results/ABLATIONS_summary.csv`,
  `results/ABLATIONS_summary.md`, `results/RUN_REPORT.md`, `results/_ablation_status.json`.

---

## 10. Diagramas SVG

Os SVGs **sem os blocos removidos** já vêm em `figures/`
(`arch_01_preprocessing_ppi_only.svg`, `arch_02_vectorial.svg`, `arch_03_image.svg`). Se não
existirem, gere equivalentes refletindo esta arquitetura (sem equalize/dilate/enhance/convexize e
sem ProtT5) e salve em `figures/*.svg`.

---

## 11. Regras de autonomia

- Trabalhe ponta a ponta; peça confirmação **apenas** nos *gates* (§6, §7) ou se faltar dado (§3).
- **Não** altere métrica, loss, base-de-treino do vetor, nem reintroduza blocos removidos.
- Corrija livremente problemas de **ambiente** (versões, download de pesos, caminhos), logando o
  que fez.
- **Nunca** reporte como completo algo parcial: logue o que rodou e o que ficou pendente.
- Ao final, garanta `results/RUN_REPORT.md` + `results/ABLATIONS_summary.{csv,md}` e liste os
  melhores wFmax(test) por ablação/domínio.

---

## 12. Checklist final

- [ ] Ambiente verificado; `data/go.obo`, CSVs e pacote `ppi_v4` presentes.
- [ ] Sanity (§6) passou; Mockups (§7) passaram (MLP, CNN1D, imagem).
- [ ] Ablações A, B, C executadas (ou BP completo + pendências logadas).
- [ ] Artefatos por run: `.pt`, `losses.png`, `metrics.{json,csv}`, `manifest.json`.
- [ ] `ABLATIONS_summary.{csv,md}` + `RUN_REPORT.md` gerados.
- [ ] Nenhum bloco removido reintroduzido; seed 1337 fixo (torch/numpy/random + DataLoader).
