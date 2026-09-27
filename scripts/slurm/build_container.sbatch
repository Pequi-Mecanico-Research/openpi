#!/bin/bash
# Constroi a imagem Apptainer do openpi em /raid/$USER/containers/openpi.sif.
# Submeter da RAIZ do repo:   sbatch scripts/slurm/build_container.sbatch
# (a pasta de logs precisa existir antes: mkdir -p /raid/$USER/vla/logs)
#SBATCH --job-name=openpi-build-sif
#SBATCH --partition=h100n3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/raid/%u/vla/logs/%x_%j.log

set -euo pipefail

RAID=${RAID:-/raid/$USER}
REPO_DIR=${REPO_DIR:-$SLURM_SUBMIT_DIR}
IMAGE=${IMAGE:-$RAID/containers/openpi.sif}

mkdir -p "$(dirname "$IMAGE")" "$RAID/vla/cache/apptainer"
# Cache/tmp do apptainer fora da home (camadas docker baixadas no build).
export APPTAINER_CACHEDIR=$RAID/vla/cache/apptainer
export APPTAINER_TMPDIR=$RAID/vla/cache/apptainer/tmp
mkdir -p "$APPTAINER_TMPDIR"

cd "$REPO_DIR"
apptainer build --fakeroot "$IMAGE" scripts/slurm/openpi.def
echo "Imagem pronta: $IMAGE"
