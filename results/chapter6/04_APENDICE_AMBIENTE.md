# Apêndice — Ambiente de execução

Documenta o hardware, as pilhas de software e os hiperparâmetros sob os quais os experimentos
foram executados. Os experimentos foram distribuídos por **cinco configurações distintas** ao longo
de duas gerações (PPI-only e campanha `dedup=max`).

**Convenção deste apêndice.** Cada tabela indica a procedência do número:

- **[M] medido** — extraído dos logs e manifests das próprias execuções, verificável nos artefatos.
- **[C] catálogo** — especificação publicada pelo fabricante, **não** medida aqui. Confira antes de
  citar na versão final; não há registro nos artefatos que a sustente.

---

## A.1 Hardware

### A.1.1 O que foi medido

| Máquina | GPU | VRAM [M] | Capacidade de computação [M] | Onde aparece |
|---|---|---|---|---|
| Servidor (nó compartilhado) | NVIDIA RTX A5500 | 23,55 GiB | — | 49 runs; blocos 1a–1d, 2a, 3d, 3e |
| Servidor (mesmo nó) | NVIDIA GeForce GTX 1080 Ti | 10,90 GiB | — | 25 runs; blocos 3a–3c |
| Notebook local | NVIDIA GeForce RTX 5060 Laptop | 8151 MiB (7,96 GiB) | **12.0** | 7 runs de CNN1D (PPI-only) |
| Google Colab | *não registrado* | *não registrado* | — | 12 runs de imagem (PPI-only) |
| Máquina do Exame de Qualificação | — (2 runs em CPU) | — | — | 14 runs (PPI-only, Ablações A e B) |

O servidor é um nó **multi-GPU compartilhado com outros usuários**, com pelo menos três placas
visíveis: índices 1 (GTX 1080 Ti), 2 (TITAN X, 11,90 GiB) e 3 (RTX A5500), sob
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. A TITAN X não recebeu nenhum experimento da campanha.

Detalhes da máquina local, consultados diretamente:

| | |
|---|---|
| CPU | Intel Core i9-14900HX — 24 núcleos (8P + 16E), 32 threads |
| RAM | 31,6 GB |
| SO | Microsoft Windows 11 Pro, build 10.0.26200 |
| Driver NVIDIA | 591.91 |

**A GPU do Colab não foi registrada.** O pipeline `ppi_only` grava `device: "cuda"` no manifest,
mas não o nome nem a VRAM da placa. O tipo (A100) consta apenas do relato da sessão. Para citar com
segurança, capture numa sessão nova e anexe a saída:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
```

### A.1.2 Especificações de catálogo — **[C], conferir antes de citar**

Nenhum número desta tabela foi medido; nenhum artefato do trabalho o sustenta. Está aqui para
contextualizar as diferenças de tempo entre máquinas.

| | RTX A5500 | GTX 1080 Ti | A100-SXM4-40GB | RTX 5060 Laptop |
|---|---|---|---|---|
| Arquitetura | Ampere (GA102) | Pascal (GP102) | Ampere (GA100) | Blackwell (GB206) |
| Capacidade de computação | 8.6 | 6.1 | 8.0 | 12.0 **[M]** |
| Núcleos CUDA | 10 240 | 3 584 | 6 912 | 3 328 |
| Núcleos Tensor | 320 (3ª ger.) | **nenhum** | 432 (3ª ger.) | 4ª ger. |
| Memória | 24 GB GDDR6 ECC | 11 GB GDDR5X | 40 GB HBM2 | 8 GB GDDR7 |
| Barramento | 384 bits | 352 bits | 5 120 bits | 128 bits |
| Banda | ~768 GB/s | ~484 GB/s | ~1 555 GB/s | ~448 GB/s |
| TDP/TGP | 230 W | 250 W | 400 W | 45–115 W (configurável) |

**Duas consequências metodológicas.**

A GTX 1080 Ti é Pascal e **não possui núcleos Tensor**. Os blocos 3a–3c rodaram nela com AMP
ativo, mas em Pascal o AMP reduz o consumo de memória sem o ganho de throughput que traria em
Ampere. Os tempos desses blocos, portanto, não são comparáveis com os da A5500 nem por regra de
três — a diferença não é só de quantidade de núcleos.

A RTX 5060 Laptop é **sm_120**, o que exige binários compilados para CUDA 12.8 ou superior. Rodas
`cu124` instalam sem erro e falham só em tempo de execução, ao lançar o primeiro kernel. Foi o
motivo de criar um ambiente separado para essa máquina.

---

## A.2 Pilhas de software

Cada run grava a própria assinatura de bibliotecas no manifest, o que permite atribuir qualquer
resultado à sua máquina de origem *a posteriori*.

### A.2.1 PPI-only — campo `libs` de `results/{tag}_manifest.json` [M]

| Runs | `device` | Python | PyTorch | torchvision | NumPy | Máquina |
|---|---|---|---|---|---|---|
| 12 | cuda | 3.12.13 | 2.11.0+cu128 | — | 2.0.2 | Colab |
| 12 | cuda | 3.10.20 | 2.4.0+cu124 | 0.19.0+cu124 | 2.2.6 | Exame de Qualificação |
| 7 | cuda | 3.11.9 | 2.11.0+cu128 | — | 1.26.4 | RTX 5060 |
| 2 | **cpu** | 3.10.20 | 2.4.0+cu124 | 0.19.0+cu124 | 2.2.6 | Exame de Qualificação |

Os dois runs em CPU são `bp_cnn1d_avg` e `bp_cnn1d_max`. A ordem de redução em CPU difere da de
GPU, então essas duas células podem divergir das demais na última casa decimal. Irrelevante para as
conclusões, que se apoiam em diferenças de 0,03 a 0,08 de wFmax, mas registrável.

### A.2.2 Campanha `dedup=max` (servidor) [M, parcial]

| | |
|---|---|
| Gerenciador | Miniconda, ambiente `ppi-gpu` |
| Caminho | `/home/f-msc2024/ra291150/miniconda/envs/ppi-gpu` |
| Python | 3.10 |
| PyTorch | ≥ 2.4 (inferido dos avisos de depreciação de `torch.cuda.amp`; **versão exata não registrada**) |
| NumPy | `<2` (imposto por `np.trapz`, removido na série 2.x e usado em `ppi_v4/metrics.py`) |
| Extra | PyTorch Geometric (blocos com `method=gnn`) |

O pipeline `ppi_v4` **não grava** um bloco `libs` no manifest, ao contrário do `ppi_only`. Para
fechar a lacuna antes da defesa:

```bash
conda activate ppi-gpu && python -c "import torch,numpy,torch_geometric as g,sys;print(sys.version);print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version());print('numpy',numpy.__version__,'pyg',g.__version__)"
```

```bash
conda list --export > ambiente_servidor.txt
```

---

## A.3 Hiperparâmetros da campanha `dedup=max` [M]

Extraídos de `tabelas/results_all.csv`; **constantes em todos os 66 runs consolidados**, o que
sustenta a leitura de que as diferenças entre blocos vêm da arquitetura e da propagação, não do
regime de treino.

| Parâmetro | Valor |
|---|---|
| `batch_size` | 256 |
| `lr` | 1 × 10⁻⁴ |
| `max_epochs` | 250 |
| `patience` | 20 |
| `seed` | 1337 |
| `thr` (limiar de aresta STRING) | 0,1 |
| `dedup` | `max` |
| `mlp_hidden_dims` | [1024, 512] |
| `mlp_dropout` | 0,4 |
| `loss_name` | `protein` |
| `pos_weight_mode` | `ic` |

**`batch_size` não é apenas hiperparâmetro do otimizador.** Em `ProteinLoss`, o termo `centric="go"`
estima o F1 de cada termo GO **ao longo da dimensão do batch**, então o tamanho do lote entra na
própria definição da perda. Com `method=gnn` ele também determina quantas propagações completas do
grafo ocorrem por época. Foi mantido fixo em 256 por essa razão.

### PPI-only

| Parâmetro | Valor |
|---|---|
| `batch_size` | 32 |
| `lr` | 1 × 10⁻⁴ |
| `max_epochs` | 200 |
| `patience` | 25 |
| `seed` | 1337 |
| `num_workers` | 0 |
| `out_size` (poolings `max`/`avg`) | 1024 |
| `image_size` | 224 |

Os regimes diferem entre as duas gerações (32/200/25 contra 256/250/20). **Não são comparáveis
entre si**, e nenhuma tabela do trabalho os cruza.

---

## A.4 Grafo STRING após deduplicação [M]

Estatísticas emitidas pelo próprio `build_sparse_graph` e gravadas em todo manifest, com
`dedup=max` e `thr=0,1`:

| Domínio | Nós | Arestas dirigidas | w mín | w máx | w médio | Nós isolados |
|---|---|---|---|---|---|---|
| bp | 92 210 | 985 840 | 0,400 | 0,999 | 0,7491 | 33 797 (36,7%) |
| cc | — | 1 133 132 | 0,400 | 0,999 | 0,7599 | 32 477 |
| mf | — | 940 412 | 0,400 | 0,999 | 0,7522 | 25 317 |

O `w_máx` de 0,999 é consequência direta da regra `max`: o escore bruto máximo do STRING no arquivo
é 167,79 e a convexização o mapeia em 0,999. Sob `dedup=sum` o mesmo par acumularia multiplicidade
(3,062 ocorrências em média, até 170), produzindo um grafo diferente — daí as duas regras darem
tabelas distintas.

A fração alta de **nós isolados** é o fato mais consequente da tabela. Na propagação fixa,
`H ← αH + (1−α)·A_T·H` com agregação nula reduz-se a `H ← αH`, de modo que a representação de um nó
isolado é multiplicada por exatamente `α^hops`, decaindo sem receber informação alguma. Isso afeta
mais de um terço dos nós em bp e cc.

---

## A.5 Custo e memória medidos [M]

### Campanha `dedup=max`, blocos 1a/1b/3a–3c (66 runs, 48,7 GPU-horas)

| Família | n | VRAM mediana | VRAM máx. | Tempo mediano | Tempo máx. | Épocas medianas |
|---|---|---|---|---|---|---|
| `fixed` + MLP | 41 | 6 544 MB | 6 603 MB | 0,32 h | 0,38 h | 243 |
| `fixed` + visão | 25 | 6 544 MB | 6 603 MB | 1,39 h | 2,00 h | **34** |

Dois pontos merecem registro. O **pico de memória é idêntico** nas duas rotas, o que indica que ele
é fixado pelos tensores da propagação (`H0` e `H_cached`), não pela cabeça de classificação. E as
**épocas até a parada antecipada diferem por um fator de 7** — a rota vetorial roda quase até o teto
de 250 épocas, enquanto a visual estabiliza por volta da 34ª. O custo maior da rota visual vem do
custo por época, não de treinar por mais tempo.

### Runs isolados de maior porte

| Run | VRAM de pico | Tempo | Máquina |
|---|---|---|---|
| `bp_gnn_gcn_mlp` | 12 952,8 MB | 4,78 h | A5500 |
| `bp_gnn_gat_mlp` | 18 864,5 MB | 7,40 h | A5500 |
| `bp_gnn_gcn_vision` | — | 1,96 h | A5500 |

A GAT consome 46% mais memória e 55% mais tempo que a GCN **já com *gradient checkpointing*
ativado** nas camadas da GNN. Sem ele, a GAT excedeu os 24 GB disponíveis.

### PPI-only

| Família | Entrada | bp | cc | mf |
|---|---|---|---|---|
| MLP `raw` | 62–74 mil | 1,13 h | 1,11 h | 0,82 h |
| CNN1D `raw` | 62–74 mil | 2,14 h | 1,76 h | 1,49 h |
| ConvNeXt-Tiny | 224×224 | 1,35–1,38 h | 1,29–1,35 h | 1,29 h |
| ResNet50 | 224×224 | 1,13–1,15 h | 1,15–1,16 h | 0,98–1,07 h |

Os tempos de imagem são do Colab e os vetoriais da RTX 5060: **não comparáveis entre si**.

---

## A.6 Notas de reprodutibilidade

**Semente única.** Todos os experimentos usam `seed=1337`, uma execução por célula. Não há
intervalo de confiança, e diferenças abaixo de ~0,005 de wFmax não são interpretáveis.

**Ordenação de dispositivos CUDA.** `CUDA_DEVICE_ORDER` tem por padrão `FASTEST_FIRST`, que **não**
corresponde à numeração do `nvidia-smi`. Num nó com placas de gerações distintas, `CUDA_VISIBLE_DEVICES=2`
pode selecionar uma placa diferente da que o `nvidia-smi` chama de 2 — foi o que ocorreu uma vez,
enviando um job para a TITAN X. Todos os scripts fixam `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

**Gradient checkpointing.** Ativado nas camadas da GNN para viabilizar a GAT em 24 GB. Verificou-se
que a operação é **numericamente transparente**: com `preserve_rng_state=True`, os pesos resultantes
são bit a bit idênticos aos obtidos sem checkpointing. Não constitui, portanto, uma diferença de
tratamento entre GCN e GAT.

**Cache de vetores (PPI-only).** `make_vector` é função pura de `(pid, rel, train_size, pooling,
out_size)`, sem RNG, e era reexecutada a cada época. O cache pré-computa os vetores uma única vez.
A equivalência foi verificada: as métricas coincidem até o ULP em float64, e duas execuções sem
cache divergem na mesma ordem de magnitude. Ganho medido de 3,9× em `cc_cnn1d_max`.

**Precisão mista.** AMP ativo em todos os runs de GPU. Nas camadas GAT e GCN os tensores de mensagem
permanecem em fp32 mesmo sob `autocast`, porque o `softmax` do PyTorch Geometric promove `exp` a
float32 — comportamento esperado, verificado por instrumentação.

**Artefatos por run.** `results/manifest_{tag}.json` (configuração, estatísticas do grafo, as seis
métricas × 2 splits, runtime), `results/probs_{tag}_{split}.npy`, `runs_v31/{tag}_loss_hist.json`,
`runs_v31/{tag}_best.pt`. A melhor época é derivada do histórico de perdas (argmin da curva de
validação), não registrada pelo laço de treino.

---

## A.7 Lacunas a fechar antes da defesa

1. **Nome e VRAM da GPU do Colab** — não registrados; o tipo A100 consta apenas do relato da sessão.
2. **Versão exata de PyTorch, CUDA, cuDNN e PyTorch Geometric no servidor** — inferiu-se apenas
   `≥ 2.4` a partir de avisos de depreciação.
3. **Especificações de catálogo da §A.1.2** — conferir junto ao fabricante; nenhuma foi medida.
4. **Modelo de CPU e RAM do servidor e do Colab** — não coletados.

Os comandos das seções A.1.1 e A.2.2 resolvem os itens 1, 2 e 4.
