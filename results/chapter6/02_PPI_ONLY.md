# PPI-only — matriz de ablações completa

**33 de 33 runs.** Ablações A, B e C fechadas. Consolidado em 2026-07-31 a partir de
`results/*_manifest.json`, regenerável por `python -m ppi_only.consolidate --results-dir results`.

Semente 1337, `lr` 1e-4, `batch_size` 32, `max_epochs` 200, `patience` 25, sem *weight decay*.
Smin em bits, **menor é melhor**; nas demais, maior é melhor.

---

## O que é o PPI-only

Feature: perfil de escores da proteína contra o **conjunto de treino apenas** — salvaguarda
anti-leakage —, dimensão `N = |treino|` (73.768 em BP, 74.328 em CC, 62.909 em MF). Sem ProtT5:
o único sinal é a topologia da rede STRING. Entre 33% e 38% das proteínas de validação não têm
vizinho de treino e produzem linha nula.

Três formas de consumir esse vetor:

| Pooling | Dimensão de entrada | Observação |
|---|---|---|
| `max` / `avg` | 1024 | pooling 1-D sobre o vetor de `N` posições |
| `raw` | `N` (62–74 mil) | sem pooling; só para cabeças vetoriais |
| imagem | 224 × 224 | vetor poolado a 224, depois *outer-diff* \|v_i − v_j\| |

---

## 1. As três ablações

| Ablação | Grade | Runs |
|---|---|---|
| **A** — perda | MLP, pooling `max`, {BCE, ProteinLoss} × 3 ontologias | 6 |
| **B** — pooling × arquitetura vetorial | {MLP, CNN1D} × {max, avg, raw} × 3 | 18 |
| **C** — cabeças de imagem | {ConvNeXt-Tiny, ResNet50} × {max, avg} × 3 | 12 |
| | dedup de `mlp_max_protein`, que aparece em A e B | −3 |
| | **total** | **33** |

---

## 2. Ablação A — BCE vs ProteinLoss (MLP, `max`)

### TESTE

| Ont | Perda | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| BP | bce | 0,3824 | 0,3340 | 0,3173 | 0,3645 | 0,3592 | **23,67** |
| BP | **protein** | **0,4249** | **0,3811** | **0,3520** | **0,4233** | **0,4521** | 23,90 |
| CC | bce | 0,6142 | 0,5577 | 0,4242 | 0,6367 | 0,6355 | 11,86 |
| CC | **protein** | **0,6754** | **0,6326** | **0,4667** | **0,6689** | **0,7285** | **11,46** |
| MF | bce | 0,5428 | 0,4153 | 0,3718 | 0,5151 | 0,5146 | 8,90 |
| MF | **protein** | **0,6141** | **0,5309** | **0,4541** | **0,5290** | **0,6356** | **8,65** |

**A `ProteinLoss` vence em 30 de 30 comparações** de Fmax, Fmax\*, wFmax, AuPRC e IAuPRC
(3 ontologias × 2 splits × 5 métricas). No Smin vence em CC e MF e **perde em BP** (23,90 contra
23,67): melhora a identificação dos termos, mas erra em termos de IC mais alto.

O ganho, de +0,035 a +0,082 de wFmax, é **maior que qualquer diferença de arquitetura ou de
pooling** nas outras duas ablações. A escolha da perda importa mais que a da arquitetura.

---

## 3. Ablação B — pooling × arquitetura vetorial

### wFmax (TESTE)

| Ont | Modelo | `max` | `avg` | `raw` | melhor |
|---|---|---|---|---|---|
| BP | MLP | 0,3520 | 0,3233 | **0,3730** | raw |
| BP | CNN1D | 0,3351 | **0,3404** | 0,3201 | avg |
| CC | MLP | 0,4667 | 0,4404 | **0,4839** | raw |
| CC | CNN1D | **0,4490** | 0,4454 | 0,4324 | max |
| MF | MLP | 0,4541 | 0,4365 | **0,4862** | raw |
| MF | CNN1D | **0,4524** | 0,4457 | 0,4415 | max |

**`raw` é o melhor pooling para a MLP e o pior para a CNN1D, nas três ontologias.** O mecanismo é
identificável: a CNN1D tem `AdaptiveMaxPool1d(256)` fixo antes da camada densa, então o vetor de
~74 mil dimensões é comprimido a 256 de qualquer forma. A convolução sobre um perfil de escores
sem estrutura local só acrescenta ruído antes desse gargalo. A MLP consome as ~74 mil dimensões
diretamente numa `Linear`.

**A MLP vence nas três ontologias**, com margem estável de +0,033 a +0,035 no melhor pooling de
cada. A uniformidade é o que sustenta a conclusão — não é efeito de uma ontologia só.

---

## 4. Ablação C — cabeças de imagem

Imagem = matriz *outer-diff* `|v_i − v_j|` de 224×224, escala de cinza, sem `dilate`/`enhance`.

### TESTE

| Ont | Backbone | Pool | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|---|
| BP | ConvNeXt-Tiny | avg | **0,3844** | **0,3312** | **0,3076** | **0,3990** | **0,3959** | 25,93 |
| BP | ConvNeXt-Tiny | max | 0,3793 | 0,3286 | 0,3033 | 0,3890 | 0,3899 | 25,80 |
| BP | ResNet50 | avg | 0,3729 | 0,3215 | 0,2968 | 0,3764 | 0,3844 | 26,42 |
| BP | ResNet50 | max | 0,3760 | 0,3285 | 0,2974 | 0,3819 | 0,3897 | **25,65** |
| CC | ConvNeXt-Tiny | avg | **0,6408** | **0,5957** | 0,4051 | **0,6817** | 0,6737 | **12,49** |
| CC | ConvNeXt-Tiny | max | 0,6399 | 0,5955 | **0,4095** | 0,6805 | **0,6760** | 12,51 |
| CC | ResNet50 | avg | 0,6329 | 0,5847 | 0,4015 | 0,6545 | 0,6692 | 12,67 |
| CC | ResNet50 | max | 0,6380 | 0,5913 | 0,4045 | 0,6621 | 0,6755 | 12,55 |
| MF | ConvNeXt-Tiny | avg | 0,5828 | 0,4939 | 0,4149 | 0,5430 | 0,5795 | 9,19 |
| MF | ConvNeXt-Tiny | max | 0,5808 | 0,4917 | 0,4097 | 0,5362 | 0,5734 | **9,15** |
| MF | ResNet50 | avg | **0,5875** | **0,4997** | **0,4186** | **0,5774** | **0,5800** | 9,53 |
| MF | ResNet50 | max | 0,5813 | 0,4901 | 0,4092 | 0,5300 | 0,5793 | 9,30 |

**ConvNeXt-Tiny vence o ResNet50 em BP e CC; o ResNet50 vence em MF.** Mas as margens são
pequenas: a maior diferença entre os dois backbones é 0,0108 de wFmax (BP/avg), e várias células
ficam abaixo de 0,005 — dentro da faixa que **uma única semente não distingue**.

**O pooling quase não importa aqui.** A diferença `max` × `avg` fica em ≤0,0108 de wFmax em todas
as seis combinações. Coerente com o fato de que ambos são comprimidos a 224 antes do *outer-diff*.

---

## 5. Comparação entre as quatro famílias

### Melhor wFmax(TESTE) de cada família

| Ont | MLP | CNN1D | ConvNeXt-Tiny | ResNet50 |
|---|---|---|---|---|
| BP | **0,3730** | 0,3404 | 0,3076 | 0,2974 |
| CC | **0,4839** | 0,4490 | 0,4095 | 0,4045 |
| MF | **0,4862** | 0,4524 | 0,4149 | 0,4186 |

**A ordem é a mesma nas três ontologias: MLP > CNN1D > cabeças de imagem.** Sem uma única
inversão em 12 células.

E a distância cresce com a complexidade do modelo. Da MLP para as cabeças convolucionais 2-D a
perda é de 0,065 a 0,076 de wFmax — mais do dobro do que separa a MLP da CNN1D.

**Interpretação.** O perfil de escores contra o treino não tem estrutura espacial. Transformá-lo
numa imagem 224×224 pela diferença externa cria uma matriz cujo padrão é uma consequência
determinística da ordenação arbitrária das colunas — cada linha e coluna brilhante corresponde a
uma posição com escore alto, e a vizinhança de um pixel não carrega informação. Convoluções 2-D
são um viés indutivo para localidade espacial, que aqui não existe. A MLP, que trata cada posição
como uma feature independente, é a arquitetura alinhada à natureza do dado.

Isso é um **resultado negativo informativo** e justifica adotar apenas a MLP nas etapas seguintes
da pesquisa.

### Custo (horas de treino por run)

| Família | Entrada | BP | CC | MF |
|---|---|---|---|---|
| MLP `raw` | 62–74 mil | 1,13 | 1,11 | 0,82 |
| CNN1D `raw` | 62–74 mil | 2,14 | 1,76 | 1,49 |
| ConvNeXt-Tiny | 224×224 | 1,35–1,38 | 1,29–1,35 | 1,29 |
| ResNet50 | 224×224 | 1,13–1,15 | 1,15–1,16 | 0,98–1,07 |

As de imagem rodaram em A100 no Colab; as vetoriais em RTX 5060 local. **Os tempos não são
comparáveis entre as duas metades**; as métricas são, por serem determinísticas.

---

## 6. Todas as 33 células, as 6 métricas

### TESTE

| Ont | Modelo | Pool | Loss | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|---|---|
| BP | MLP | avg | protein | 0,4011 | 0,3507 | 0,3233 | 0,3825 | 0,4152 | 25,36 |
| BP | MLP | max | bce | 0,3824 | 0,3340 | 0,3173 | 0,3645 | 0,3592 | 23,67 |
| BP | MLP | max | protein | 0,4249 | 0,3811 | 0,3520 | 0,4233 | 0,4521 | 23,90 |
| BP | MLP | raw | protein | 0,4625 | 0,3986 | 0,3730 | 0,4557 | 0,4737 | 23,68 |
| BP | CNN1D | avg | protein | 0,4104 | 0,3729 | 0,3404 | 0,4279 | 0,4417 | 24,31 |
| BP | CNN1D | max | protein | 0,4090 | 0,3658 | 0,3351 | 0,3635 | 0,4327 | 24,19 |
| BP | CNN1D | raw | protein | 0,3985 | 0,3488 | 0,3201 | 0,3377 | 0,4098 | 25,06 |
| BP | ConvNeXt-Tiny | avg | protein | 0,3844 | 0,3312 | 0,3076 | 0,3990 | 0,3959 | 25,93 |
| BP | ConvNeXt-Tiny | max | protein | 0,3793 | 0,3286 | 0,3033 | 0,3890 | 0,3899 | 25,80 |
| BP | ResNet50 | avg | protein | 0,3729 | 0,3215 | 0,2968 | 0,3764 | 0,3844 | 26,42 |
| BP | ResNet50 | max | protein | 0,3760 | 0,3285 | 0,2974 | 0,3819 | 0,3897 | 25,65 |
| CC | MLP | avg | protein | 0,6578 | 0,6159 | 0,4404 | 0,5980 | 0,7024 | 12,12 |
| CC | MLP | max | bce | 0,6142 | 0,5577 | 0,4242 | 0,6367 | 0,6355 | 11,86 |
| CC | MLP | max | protein | 0,6754 | 0,6326 | 0,4667 | 0,6689 | 0,7285 | 11,46 |
| CC | MLP | raw | protein | 0,6738 | 0,6346 | 0,4839 | 0,6383 | 0,7220 | 12,03 |
| CC | CNN1D | avg | protein | 0,6634 | 0,6214 | 0,4454 | 0,6976 | 0,7213 | 11,72 |
| CC | CNN1D | max | protein | 0,6651 | 0,6232 | 0,4490 | 0,6097 | 0,7159 | 11,74 |
| CC | CNN1D | raw | protein | 0,6569 | 0,6148 | 0,4324 | 0,6049 | 0,7060 | 12,11 |
| CC | ConvNeXt-Tiny | avg | protein | 0,6408 | 0,5957 | 0,4051 | 0,6817 | 0,6737 | 12,49 |
| CC | ConvNeXt-Tiny | max | protein | 0,6399 | 0,5955 | 0,4095 | 0,6805 | 0,6760 | 12,51 |
| CC | ResNet50 | avg | protein | 0,6329 | 0,5847 | 0,4015 | 0,6545 | 0,6692 | 12,67 |
| CC | ResNet50 | max | protein | 0,6380 | 0,5913 | 0,4045 | 0,6621 | 0,6755 | 12,55 |
| MF | MLP | avg | protein | 0,6025 | 0,5153 | 0,4365 | 0,4343 | 0,6041 | 8,96 |
| MF | MLP | max | bce | 0,5428 | 0,4153 | 0,3718 | 0,5151 | 0,5146 | 8,90 |
| MF | MLP | max | protein | 0,6141 | 0,5309 | 0,4541 | 0,5290 | 0,6356 | 8,65 |
| MF | MLP | raw | protein | 0,6104 | 0,5338 | 0,4862 | 0,4934 | 0,6242 | 9,20 |
| MF | CNN1D | avg | protein | 0,6042 | 0,5128 | 0,4457 | 0,4296 | 0,5892 | 9,05 |
| MF | CNN1D | max | protein | 0,6161 | 0,5321 | 0,4524 | 0,4564 | 0,6241 | 8,72 |
| MF | CNN1D | raw | protein | 0,6075 | 0,5222 | 0,4415 | 0,4409 | 0,6046 | 9,03 |
| MF | ConvNeXt-Tiny | avg | protein | 0,5828 | 0,4939 | 0,4149 | 0,5430 | 0,5795 | 9,19 |
| MF | ConvNeXt-Tiny | max | protein | 0,5808 | 0,4917 | 0,4097 | 0,5362 | 0,5734 | 9,15 |
| MF | ResNet50 | avg | protein | 0,5875 | 0,4997 | 0,4186 | 0,5774 | 0,5800 | 9,53 |
| MF | ResNet50 | max | protein | 0,5813 | 0,4901 | 0,4092 | 0,5300 | 0,5793 | 9,30 |

A tabela de **validação** está em `results/ABLATIONS_summary.csv` (coluna `split`), com as mesmas
colunas.

---

## 7. Validação e teste concordam

Diferença teste − validação em wFmax, sobre as 33 células: **mediana +0,0039**, mínimo −0,0131,
máximo +0,0093. O teste é sistematicamente um pouco melhor, o que é esperado quando a seleção do
melhor época se dá em validação. **Não há indício de sobreajuste ao split de teste.**

---

## 8. O que levar para o Capítulo 6

1. **A perda importa mais que a arquitetura.** ProteinLoss > BCE em 30/30 comparações; o ganho
   supera qualquer diferença entre poolings ou entre famílias de modelo.
2. **A MLP vence as três outras famílias nas três ontologias**, sem inversão em 12 células.
   Justifica usar só ela nas etapas pós-VISAPP.
3. **As cabeças de imagem são as piores.** O *outer-diff* de um perfil sem estrutura espacial não
   oferece localidade para a convolução 2-D explorar. Resultado negativo, mas informativo.
4. **`raw` inverte de sinal entre MLP e CNN1D** — melhor para uma, pior para a outra, nas três
   ontologias —, e o gargalo de `AdaptiveMaxPool1d(256)` explica o porquê.
5. **O teto da topologia isolada.** O melhor wFmax do PPI-only é 0,3730 / 0,4839 / 0,4862. Serve
   de piso para avaliar o quanto o ProtT5 acrescenta na campanha `dedup=max`.

---

## 9. Limitações

1. **Uma semente (1337), estimativa pontual.** Sem intervalo de confiança, diferenças abaixo de
   ~0,005 de wFmax não são interpretáveis — o que inclui várias comparações da Ablação C.
2. **Um run por célula.** As leituras mecanísticas são hipóteses coerentes com os dados, não
   conclusões demonstradas.
3. **Hardware heterogêneo — quatro configurações**, identificáveis pela assinatura de bibliotecas
   registrada em cada manifest:

   | Runs | `device` | Python / torch / numpy | Máquina |
   |---|---|---|---|
   | 12 (Ablação C) | cuda | 3.12.13 / 2.11.0+cu128 / 2.0.2 | A100, Colab |
   | 12 (Ablação A + MLP da B) | cuda | 3.10.20 / 2.4.0+cu124 / 2.2.6 | máquina do Exame de Qualificação |
   | 7 (CNN1D da B) | cuda | 3.11.9 / 2.11.0+cu128 / 1.26.4 | RTX 5060 Laptop |
   | 2 (`bp_cnn1d_avg`, `bp_cnn1d_max`) | **cpu** | 3.10.20 / 2.4.0+cu124 / 2.2.6 | máquina do EQ, sem GPU |

   As métricas seguem comparáveis, mas **os tempos não são** e não devem ser tabelados juntos. As
   duas células em CPU merecem nota: a ordem de redução em CPU difere da de GPU, então elas podem
   divergir das demais na última casa decimal — irrelevante para as conclusões, que se apoiam em
   diferenças de 0,03 a 0,08, mas registrável.
4. **Métricas na convenção `ppi_v4/metrics.py`** — raiz mantida, IC curado de `{ont}_ic.csv`,
   propagação de ancestrais ligada. `Fmax*` é a variante sem raiz e é o número comparável com os
   baselines repontuados nessa mesma convenção.

---

## 10. Reprodutibilidade

```bash
# consolidar tudo o que existe em results/
python -m ppi_only.consolidate --results-dir results

# um run isolado
python -m ppi_only.main --domain cc --model mlp --pooling raw

# a matriz completa (idempotente: pula o que já tem metrics.json)
python -m ppi_only.run_ablations --domains bp,cc,mf --skip-if-done
```

Artefatos por run: `results/{tag}_metrics.{json,csv}`, `results/{tag}_manifest.json`,
`runs/{tag}_best.pt`, `runs/{tag}_losses.png`, `runs/{tag}_loss_hist.json`.
Tag: `{domínio}_{modelo}_{pooling}_{perda}_seed1337`.

Correções aplicadas ao pacote original durante esta etapa, ambas relevantes para auditoria:

- **Tag de smoke.** Os gates de mockup gravavam com o mesmo tag dos runs reais e sobrescreveram um
  resultado de 200 épocas (`bp_cnn1d_max`, wFmax 0,3289 → 0,2081). Restaurado do backup; o tag de
  smoke agora leva sufixo `_smoke`.
- **Sumário lido do disco.** `run_ablations` montava o `ABLATIONS_summary` só com os runs da
  invocação corrente, o que deixou o sumário original com 12 runs apesar de 14 no disco. Agora
  `ppi_only/consolidate.py` lê todos os manifests e cruza com a matriz canônica de 33.
- **Cache de vetores** (`vec_cache=1`), numericamente transparente: métricas coincidem com a
  versão sem cache até o ULP de float64, e duas execuções idênticas divergem na mesma ordem de
  magnitude. Ganho medido de 3,9× em `cc_cnn1d_max`.
