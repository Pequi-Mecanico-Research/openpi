#!/bin/bash
# Constroi a imagem Apptainer do openpi em /raid/$USER/teleop/containers/openpi.sif.
# Submeter da RAIZ do repo:   sbatch scripts/slurm/build_container.sbatch
# (a pasta de logs precisa existir antes: mkdir -p /raid/$USER/teleop/vla/logs)
#SBATCH --job-name=openpi-build-sif
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/raid/%u/teleop/vla/logs/%x_%j.log

set -euo pipefail

RAID=${RAID:-/raid/$USER}
TELEOP_DIR=${TELEOP_DIR:-$RAID/teleop}
REPO_DIR=${REPO_DIR:-$SLURM_SUBMIT_DIR}
IMAGE=${IMAGE:-$TELEOP_DIR/containers/openpi.sif}

mkdir -p "$(dirname "$IMAGE")" "$TELEOP_DIR/vla/cache/apptainer"
# Cache/tmp do apptainer fora da home (camadas docker baixadas no build).
export APPTAINER_CACHEDIR=$TELEOP_DIR/vla/cache/apptainer
export APPTAINER_TMPDIR=$TELEOP_DIR/vla/cache/apptainer/tmp
mkdir -p "$APPTAINER_TMPDIR"

cd "$REPO_DIR"
apptainer build --fakeroot "$IMAGE" scripts/slurm/openpi.def
echo "Imagem pronta: $IMAGE"
