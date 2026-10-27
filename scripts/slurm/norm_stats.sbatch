#!/bin/bash
# Calcula as norm stats do dataset para o config (grava em assets/<config>/<repo_id>/norm_stats.json dentro do repo).
# Tambem baixa o dataset do HF para o cache no /raid e valida o decode dos videos AV1 antes de alocar GPU no treino.
# Submeter da RAIZ do repo:   sbatch scripts/slurm/norm_stats.sbatch
#SBATCH --job-name=openpi-norm-stats
#SBATCH --partition=h100n3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/raid/%u/vla/logs/%x_%j.log

set -euo pipefail
source "${REPO_DIR:-$SLURM_SUBMIT_DIR}/scripts/slurm/env.sh"

echo "Config: $CONFIG_NAME | repo: $REPO_DIR | node: $(hostname)"
# Sem GPU neste job: o JAX cai para CPU, o que basta para as estatisticas.
run_in_container python scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
find "$REPO_DIR/assets/$CONFIG_NAME" -name norm_stats.json -print
