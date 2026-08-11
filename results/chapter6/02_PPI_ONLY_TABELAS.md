# PPI-only — as três ablações, validação e teste

33 runs. Sem ProtT5: o único sinal é a topologia da rede, o que faz deste bloco o **piso** contra o qual a campanha é medida.

Convenção `ppi_v4/metrics.py` — raiz mantida, IC curado de `{ont}_ic.csv`, propagação de ancestrais ligada. Smin em bits, **menor é melhor**.

As decisões de projeto se apoiam em **validação**; o teste aparece para o confronto final.

---

# VALIDAÇÃO

## Ablação A — BCE vs ProteinLoss (MLP, pooling `max`)

Isola o efeito da função de perda, tudo o mais constante.

| Ont | Perda | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| BP | `bce` | 0,3786 | 0,3287 | 0,3119 | 0,3588 | 0,3536 | 23,21 |
| BP | `protein` | 0,4213 | 0,3759 | 0,3467 | 0,4175 | 0,4476 | 23,32 |
| CC | `bce` | 0,6140 | 0,5563 | 0,4201 | 0,6356 | 0,6347 | 11,71 |
| CC | `protein` | 0,6748 | 0,6310 | 0,4608 | 0,6674 | 0,7271 | 11,40 |
| MF | `bce` | 0,5445 | 0,4177 | 0,3785 | 0,5138 | 0,5134 | 8,98 |
| MF | `protein` | 0,6212 | 0,5401 | 0,4672 | 0,5336 | 0,6426 | 8,62 |

## Ablação B — pooling × arquitetura vetorial

`raw` entrega o perfil inteiro (62–74 mil dimensões); `max`/`avg` comprimem a 1024 antes da rede.

| Ont | Modelo | Pooling | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|---|
| BP | MLP | `max` | 0,4213 | 0,3759 | 0,3467 | 0,4175 | 0,4476 | 23,32 |
| BP | MLP | `avg` | 0,3980 | 0,3459 | 0,3188 | 0,3770 | 0,4116 | 24,87 |
| BP | MLP | `raw` | 0,4581 | 0,3897 | 0,3637 | 0,4481 | 0,4675 | 23,14 |
| BP | CNN1D | `max` | 0,4058 | 0,3588 | 0,3289 | 0,3564 | 0,4276 | 23,78 |
| BP | CNN1D | `avg` | 0,4066 | 0,3684 | 0,3333 | 0,4231 | 0,4378 | 23,79 |
| BP | CNN1D | `raw` | 0,3943 | 0,3434 | 0,3147 | 0,3311 | 0,4045 | 24,64 |
| CC | MLP | `max` | 0,6748 | 0,6310 | 0,4608 | 0,6674 | 0,7271 | 11,40 |
| CC | MLP | `avg` | 0,6582 | 0,6154 | 0,4355 | 0,5970 | 0,7029 | 11,96 |
| CC | MLP | `raw` | 0,6741 | 0,6345 | 0,4825 | 0,6351 | 0,7208 | 11,96 |
| CC | CNN1D | `max` | 0,6678 | 0,6256 | 0,4433 | 0,6098 | 0,7166 | 11,62 |
| CC | CNN1D | `avg` | 0,6647 | 0,6223 | 0,4393 | 0,6992 | 0,7218 | 11,65 |
| CC | CNN1D | `raw` | 0,6580 | 0,6154 | 0,4281 | 0,6044 | 0,7059 | 12,03 |
| MF | MLP | `max` | 0,6212 | 0,5401 | 0,4672 | 0,5336 | 0,6426 | 8,62 |
| MF | MLP | `avg` | 0,6053 | 0,5198 | 0,4452 | 0,4369 | 0,6087 | 8,99 |
| MF | MLP | `raw` | 0,6135 | 0,5369 | 0,4892 | 0,4955 | 0,6273 | 9,21 |
| MF | CNN1D | `max` | 0,6196 | 0,5364 | 0,4627 | 0,4574 | 0,6264 | 8,74 |
| MF | CNN1D | `avg` | 0,6049 | 0,5135 | 0,4509 | 0,4290 | 0,5913 | 9,15 |
| MF | CNN1D | `raw` | 0,6122 | 0,5282 | 0,4511 | 0,4426 | 0,6086 | 9,06 |

## Ablação C — cabeças de imagem

O vetor vira uma matriz 224×224 pela diferença externa |v_i − v_j|.

| Ont | Backbone | Pooling | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|---|
| BP | ConvNeXt-Tiny | `max` | 0,3765 | 0,3222 | 0,2966 | 0,3836 | 0,3852 | 25,30 |
| BP | ConvNeXt-Tiny | `avg` | 0,3810 | 0,3263 | 0,3040 | 0,3948 | 0,3921 | 25,41 |
| BP | ResNet50 | `max` | 0,3700 | 0,3209 | 0,2905 | 0,3754 | 0,3826 | 25,10 |
| BP | ResNet50 | `avg` | 0,3677 | 0,3141 | 0,2894 | 0,3674 | 0,3760 | 25,87 |
| CC | ConvNeXt-Tiny | `max` | 0,6413 | 0,5970 | 0,4084 | 0,6821 | 0,6772 | 12,34 |
| CC | ConvNeXt-Tiny | `avg` | 0,6432 | 0,5978 | 0,4038 | 0,6854 | 0,6785 | 12,33 |
| CC | ResNet50 | `max` | 0,6403 | 0,5935 | 0,4006 | 0,6623 | 0,6771 | 12,45 |
| CC | ResNet50 | `avg` | 0,6337 | 0,5853 | 0,3978 | 0,6569 | 0,6718 | 12,52 |
| MF | ConvNeXt-Tiny | `max` | 0,5851 | 0,4958 | 0,4172 | 0,5379 | 0,5791 | 9,17 |
| MF | ConvNeXt-Tiny | `avg` | 0,5852 | 0,4969 | 0,4208 | 0,5451 | 0,5829 | 9,27 |
| MF | ResNet50 | `max` | 0,5858 | 0,4956 | 0,4187 | 0,5322 | 0,5861 | 9,29 |
| MF | ResNet50 | `avg` | 0,5888 | 0,5015 | 0,4242 | 0,5766 | 0,5806 | 9,61 |

## Melhor wFmax de cada família — VALIDAÇÃO

| Ont | MLP | CNN1D | ConvNeXt-Tiny | ResNet50 |
|---|---|---|---|---|
| BP | **0,3637** | 0,3333 | 0,3040 | 0,2905 |
| CC | **0,4825** | 0,4433 | 0,4084 | 0,4006 |
| MF | **0,4892** | 0,4627 | 0,4208 | 0,4242 |

---

# TESTE

## Ablação A — BCE vs ProteinLoss (MLP, pooling `max`)

Isola o efeito da função de perda, tudo o mais constante.

| Ont | Perda | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|
| BP | `bce` | 0,3824 | 0,3340 | 0,3173 | 0,3645 | 0,3592 | 23,67 |
| BP | `protein` | 0,4249 | 0,3811 | 0,3520 | 0,4233 | 0,4521 | 23,90 |
| CC | `bce` | 0,6142 | 0,5577 | 0,4242 | 0,6367 | 0,6355 | 11,86 |
| CC | `protein` | 0,6754 | 0,6326 | 0,4667 | 0,6689 | 0,7285 | 11,46 |
| MF | `bce` | 0,5428 | 0,4153 | 0,3718 | 0,5151 | 0,5146 | 8,90 |
| MF | `protein` | 0,6141 | 0,5309 | 0,4541 | 0,5290 | 0,6356 | 8,65 |

## Ablação B — pooling × arquitetura vetorial

`raw` entrega o perfil inteiro (62–74 mil dimensões); `max`/`avg` comprimem a 1024 antes da rede.

| Ont | Modelo | Pooling | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
|---|---|---|---|---|---|---|---|---|
| BP | MLP | `max` | 0,4249 | 0,3811 | 0,3520 | 0,4233 | 0,4521 | 23,90 |
| BP | MLP | `avg` | 0,4011 | 0,3507 | 0,3233 | 0,3825 | 0,4152 | 25,36 |
| BP | MLP | `raw` | 0,4625 | 0,3986 | 0,3730 | 0,4557 | 0,4737 | 23,68 |
| BP | CNN1D | `max` | 0,4090 | 0,3658 | 0,3351 | 0,3635 | 0,4327 | 24,19 |
| BP | CNN1D | `avg` | 0,4104 | 0,3729 | 0,3404 | 0,4279 | 0,4417 | 24,31 |
| BP | CNN1D | `raw` | 0,3985 | 0,3488 | 0,3201 | 0,3377 | 0,4098 | 25,06 |
| CC | MLP | `max` | 0,6754 | 0,6326 | 0,4667 | 0,6689 | 0,7285 | 11,46 |
| CC | MLP | `avg` | 0,6578 | 0,6159 | 0,4404 | 0,5980 | 0,7024 | 12,12 |
| CC | MLP | `raw` | 0,6738 | 0,6346 | 0,4839 | 0,6383 | 0,7220 | 12,03 |
| CC | CNN1D | `max` | 0,6651 | 0,6232 | 0,4490 | 0,6097 | 0,7159 | 11,74 |
| CC | CNN1D | `avg` | 0,6634 | 0,6214 | 0,4454 | 0,6976 | 0,7213 | 11,72 |
| CC | CNN1D | `raw` | 0,6569 | 0,6148 | 0,4324 | 0,6049 | 0,7060 | 12,11 |
| MF | MLP | `max` | 0,6141 | 0,5309 | 0,4541 | 0,5290 | 0,6356 | 8,65 |
| MF | MLP | `avg` | 0,6025 | 0,5153 | 0,4365 | 0,4343 | 0,6041 | 8,96 |
| MF | MLP | `raw` | 0,6104 | 0,5338 | 0,4862 | 0,4934 | 0,6242 | 9,20 |
| MF | CNN1D | `max` | 0,6161 | 0,5321 | 0,4524 | 0,4564 | 0,6241 | 8,72 |
| MF | CNN1D | `avg` | 0,6042 | 0,5128 | 0,4457 | 0,4296 | 0,5892 | 9,05 |
| MF | CNN1D | `raw` | 0,6075 | 0,5222 | 0,4415 | 0,4409 | 0,6046 | 9,03 |

## Ablação C — cabeças de imagem

O vetor vira uma matriz 224×224 pela diferença externa |v_i − v_j|.

| Ont | Backbone | Pooling | Fmax | Fmax* | wFmax | AuPRC | IAuPRC | Smin ↓ |
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

## Melhor wFmax de cada família — TESTE

| Ont | MLP | CNN1D | ConvNeXt-Tiny | ResNet50 |
|---|---|---|---|---|
| BP | **0,3730** | 0,3404 | 0,3076 | 0,2974 |
| CC | **0,4839** | 0,4490 | 0,4095 | 0,4045 |
| MF | **0,4862** | 0,4524 | 0,4149 | 0,4186 |

---

## Campeão de cada ontologia

Escolhido pelo wFmax de **validação**; as duas colunas de wFmax mostram os dois splits do mesmo run.

| Ont | Run | wFmax val | wFmax teste |
|---|---|---|---|
| BP | `bp_mlp_raw_protein_seed1337` | **0,3637** | 0,3730 |
| CC | `cc_mlp_raw_protein_seed1337` | **0,4825** | 0,4839 |
| MF | `mf_mlp_raw_protein_seed1337` | **0,4892** | 0,4862 |
