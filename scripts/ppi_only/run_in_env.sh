#!/usr/bin/env bash
# Activate the deepgraphgo conda env and run the given command from the projeto/ dir.
# Usage: bash run_in_env.sh <command> [args...]
source /home/yeonatan/miniconda3/etc/profile.d/conda.sh
conda activate deepgraphgo || { echo "!! conda activate deepgraphgo failed"; exit 97; }
cd "/home/yeonatan/Área de trabalho/projeto-ppi-only-baixar-20260724T213600Z-1-001/ppi-only/projeto" || { echo "!! cd to projeto failed"; exit 98; }
exec "$@"
