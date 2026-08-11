# Resumo das Ablações (PPI-only)

Consolidado de **33 runs** em `results/`. Métrica de mérito: **wFmax** (teste). As 6 métricas por run em `*_metrics.csv`.


## Ablação A — MLP: BCE vs ProteinLoss

| domínio | modelo | pooling | loss | wFmax(test) | Fmax | Fmax* | Smin | AUPRC | iAUPRC |
|---|---|---|---|---|---|---|---|---|---|
| bp | mlp | max | bce | 0.3173 | 0.3824 | 0.3340 | 23.6703 | 0.3645 | 0.3592 |
| bp | mlp | max | protein | 0.3520 | 0.4249 | 0.3811 | 23.9044 | 0.4233 | 0.4521 |
| cc | mlp | max | bce | 0.4242 | 0.6142 | 0.5577 | 11.8599 | 0.6367 | 0.6355 |
| cc | mlp | max | protein | 0.4667 | 0.6754 | 0.6326 | 11.4602 | 0.6689 | 0.7285 |
| mf | mlp | max | bce | 0.3718 | 0.5428 | 0.4153 | 8.9039 | 0.5151 | 0.5146 |
| mf | mlp | max | protein | 0.4541 | 0.6141 | 0.5309 | 8.6531 | 0.5290 | 0.6356 |

**Melhor wFmax(test):** 0.4667 (cc/mlp/max/protein)


## Ablação B — MLP/CNN1D: pooling Max/Avg/Raw

| domínio | modelo | pooling | loss | wFmax(test) | Fmax | Fmax* | Smin | AUPRC | iAUPRC |
|---|---|---|---|---|---|---|---|---|---|
| bp | cnn1d | avg | protein | 0.3404 | 0.4104 | 0.3729 | 24.3105 | 0.4279 | 0.4417 |
| bp | cnn1d | max | protein | 0.3351 | 0.4090 | 0.3658 | 24.1884 | 0.3635 | 0.4327 |
| bp | cnn1d | raw | protein | 0.3201 | 0.3985 | 0.3488 | 25.0647 | 0.3377 | 0.4098 |
| bp | mlp | avg | protein | 0.3233 | 0.4011 | 0.3507 | 25.3614 | 0.3825 | 0.4152 |
| bp | mlp | max | protein | 0.3520 | 0.4249 | 0.3811 | 23.9044 | 0.4233 | 0.4521 |
| bp | mlp | raw | protein | 0.3730 | 0.4625 | 0.3986 | 23.6756 | 0.4557 | 0.4737 |
| cc | cnn1d | avg | protein | 0.4454 | 0.6634 | 0.6214 | 11.7187 | 0.6976 | 0.7213 |
| cc | cnn1d | max | protein | 0.4490 | 0.6651 | 0.6232 | 11.7450 | 0.6097 | 0.7159 |
| cc | cnn1d | raw | protein | 0.4324 | 0.6569 | 0.6148 | 12.1070 | 0.6049 | 0.7060 |
| cc | mlp | avg | protein | 0.4404 | 0.6578 | 0.6159 | 12.1195 | 0.5980 | 0.7024 |
| cc | mlp | max | protein | 0.4667 | 0.6754 | 0.6326 | 11.4602 | 0.6689 | 0.7285 |
| cc | mlp | raw | protein | 0.4839 | 0.6738 | 0.6346 | 12.0340 | 0.6383 | 0.7220 |
| mf | cnn1d | avg | protein | 0.4457 | 0.6042 | 0.5128 | 9.0507 | 0.4296 | 0.5892 |
| mf | cnn1d | max | protein | 0.4524 | 0.6161 | 0.5321 | 8.7225 | 0.4564 | 0.6241 |
| mf | cnn1d | raw | protein | 0.4415 | 0.6075 | 0.5222 | 9.0339 | 0.4409 | 0.6046 |
| mf | mlp | avg | protein | 0.4365 | 0.6025 | 0.5153 | 8.9614 | 0.4343 | 0.6041 |
| mf | mlp | max | protein | 0.4541 | 0.6141 | 0.5309 | 8.6531 | 0.5290 | 0.6356 |
| mf | mlp | raw | protein | 0.4862 | 0.6104 | 0.5338 | 9.1972 | 0.4934 | 0.6242 |

**Melhor wFmax(test):** 0.4862 (mf/mlp/raw/protein)


## Ablação C — ResNet50/ConvNeXt-Tiny: pooling Max/Avg

| domínio | modelo | pooling | loss | wFmax(test) | Fmax | Fmax* | Smin | AUPRC | iAUPRC |
|---|---|---|---|---|---|---|---|---|---|
| bp | convnext_tiny | avg | protein | 0.3076 | 0.3844 | 0.3312 | 25.9292 | 0.3990 | 0.3959 |
| bp | convnext_tiny | max | protein | 0.3033 | 0.3793 | 0.3286 | 25.8030 | 0.3890 | 0.3899 |
| bp | resnet50 | avg | protein | 0.2968 | 0.3729 | 0.3215 | 26.4223 | 0.3764 | 0.3844 |
| bp | resnet50 | max | protein | 0.2974 | 0.3760 | 0.3285 | 25.6518 | 0.3819 | 0.3897 |
| cc | convnext_tiny | avg | protein | 0.4051 | 0.6408 | 0.5957 | 12.4899 | 0.6817 | 0.6737 |
| cc | convnext_tiny | max | protein | 0.4095 | 0.6399 | 0.5955 | 12.5107 | 0.6805 | 0.6760 |
| cc | resnet50 | avg | protein | 0.4015 | 0.6329 | 0.5847 | 12.6690 | 0.6545 | 0.6692 |
| cc | resnet50 | max | protein | 0.4045 | 0.6380 | 0.5913 | 12.5541 | 0.6621 | 0.6755 |
| mf | convnext_tiny | avg | protein | 0.4149 | 0.5828 | 0.4939 | 9.1932 | 0.5430 | 0.5795 |
| mf | convnext_tiny | max | protein | 0.4097 | 0.5808 | 0.4917 | 9.1452 | 0.5362 | 0.5734 |
| mf | resnet50 | avg | protein | 0.4186 | 0.5875 | 0.4997 | 9.5327 | 0.5774 | 0.5800 |
| mf | resnet50 | max | protein | 0.4092 | 0.5813 | 0.4901 | 9.3019 | 0.5300 | 0.5793 |

**Melhor wFmax(test):** 0.4186 (mf/resnet50/avg/protein)
