#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 

DATA_DIR="data"
RAW_DIR="raw/raw_bp"
PPI_CSV="data/ppi.csv"

echo "========================================="
echo "INICIANDO BATERIA DE EXPERIMENTOS CAFA5"
echo "========================================="

echo "[1/6] Rodando Fixo + Vision (ConvNeXt)..."
python -m ppi_v31.main_v32 --config configs/01_fixo_vision.yaml \
  --data-dir $DATA_DIR --raw-dir $RAW_DIR --ppi-csv $PPI_CSV

echo "[2/6] Rodando Fixo + MLP..."
python -m ppi_v31.main_v32 --config configs/02_fixo_mlp.yaml \
  --data-dir $DATA_DIR --raw-dir $RAW_DIR --ppi-csv $PPI_CSV

echo "[3/6] Rodando Treino da GNN Pura (Isso vai salvar o arquivo .pt!)..."
python -m ppi_v31.main_v32 --config configs/03_gnn_puro.yaml \
  --data-dir $DATA_DIR --raw-dir $RAW_DIR --ppi-csv $PPI_CSV

echo "[4/6] Rodando Embeddings da GNN + Vision (ConvNeXt)..."
python -m ppi_v31.main_v32 --config configs/04_load_gnn_vision.yaml \
  --data-dir $DATA_DIR --raw-dir $RAW_DIR --ppi-csv $PPI_CSV

echo "[5/6] Rodando Fusão Híbrida (Add) + Vision..."
python -m ppi_v31.main_v32 --config configs/05_hybrid_add_vision.yaml \
  --data-dir $DATA_DIR --raw-dir $RAW_DIR --ppi-csv $PPI_CSV

echo "[6/6] Rodando Fusão Híbrida (Concat) + MLP..."
python -m ppi_v31.main_v32 --config configs/06_hybrid_concat_mlp.yaml \
  --data-dir $DATA_DIR --raw-dir $RAW_DIR --ppi-csv $PPI_CSV

echo "========================================="
echo "TODOS OS EXPERIMENTOS CONCLUÍDOS COM SUCESSO!"
echo "Verifique a pasta runs_v31/ para os arquivos JSON com os resultados de wFmax e Smin."
echo "========================================="