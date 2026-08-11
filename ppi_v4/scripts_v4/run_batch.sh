#!/bin/bash

# Garante a ordem correta das GPUs e trava na GPU 2 (RTX A5500 vazia)
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1

echo "======================================================"
echo " INICIANDO BATERIA DE ABLAÇÕES (PULANDO GNNs PURAS) "
echo "======================================================"

# Lista exata dos arquivos da sua pasta configs, sem os 03
CONFIGS=(
    "01_fixo_vision.yaml"
    "02_fixo_mlp.yaml"
    "04_hibrido_mlp.yaml"
    "04_load_gnn_vision.yaml"
    "05_hybrid_add_vision.yaml"
    "06_hybrid_concat_mlp.yaml"
)

# Laço que vai rodar um por um
for conf in "${CONFIGS[@]}"; do
    # Extrai o nome base (tira o .yaml)
    basename="${conf%.yaml}"

    # Regra: Se o arquivo começar com 04, 05 ou 06 (Híbridos), adiciona _gat_fc
    if [[ "$conf" == 04* ]] || [[ "$conf" == 05* ]] || [[ "$conf" == 06* ]]; then
        logfile="log_${basename}_gat_fc.txt"
    else
        logfile="log_${basename}.txt"
    fi

    echo "[$(date +'%H:%M:%S')] Iniciando: $conf"
    echo " -> Salvando log em: $logfile"

    # Executa o modelo
    python -m ppi_v4.main_v32 \
        --config "configs/${conf}" \
        --data-dir data \
        --ppi-csv data/ppi.csv > "$logfile" 2>&1

    echo "[$(date +'%H:%M:%S')] Concluído: $conf"
    echo "------------------------------------------------------"
done

echo "======================================================"
echo " TODOS OS EXPERIMENTOS FORAM FINALIZADOS COM SUCESSO! "
echo "======================================================"