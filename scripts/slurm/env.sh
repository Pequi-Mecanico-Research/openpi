# Configuracao comum dos jobs Slurm do openpi. Carregado via `source` pelos .sbatch.
# Regra do cluster: nada de dados/caches na home -> tudo abaixo fica em /raid/$USER.
# Qualquer variavel pode ser sobrescrita no sbatch: `sbatch --export=ALL,EXP_NAME=teste scripts/slurm/train_full.sbatch`.

RAID=${RAID:-/raid/$USER}
PROJECT_DIR=${PROJECT_DIR:-$RAID/vla}
# Os .sbatch devem ser submetidos da raiz do repo (clonado dentro do /raid): o Slurm copia o script para um
# diretorio de spool, entao o caminho do repo vem do diretorio de submissao.
REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
IMAGE=${IMAGE:-$RAID/containers/openpi.sif}
CKPT_BASE=${CKPT_BASE:-$RAID/checkpoints/openpi}
CACHE_DIR=${CACHE_DIR:-$PROJECT_DIR/cache}
# Home "falsa" do container: ~/.cache, ~/.netrc (login do wandb), ~/.config, etc. ficam no /raid.
CONTAINER_HOME=${CONTAINER_HOME:-$PROJECT_DIR/home}

CONFIG_NAME=${CONFIG_NAME:-pi05_widowxai_organize_table_full}

mkdir -p "$PROJECT_DIR/logs" "$CKPT_BASE" "$CACHE_DIR"/{hf,openpi,jax,wandb,tmp,xdg} "$CONTAINER_HOME"

if [[ ! -f "$IMAGE" ]]; then
    echo "ERRO: imagem $IMAGE nao encontrada. Rode scripts/slurm/build_container.sbatch primeiro." >&2
    exit 1
fi
if [[ ! -f "$REPO_DIR/scripts/train.py" ]]; then
    echo "ERRO: repo openpi nao encontrado em $REPO_DIR." >&2
    exit 1
fi

# Executa um comando dentro do container, com o repo montado em /app e o /raid no mesmo caminho.
# --cleanenv evita vazar PYTHONPATH/conda do host; o que o treino precisa e passado explicitamente.
run_in_container() {
    local secrets=()
    [[ -n "${WANDB_API_KEY:-}" ]] && secrets+=(--env "WANDB_API_KEY=$WANDB_API_KEY")
    [[ -n "${HF_TOKEN:-}" ]] && secrets+=(--env "HF_TOKEN=$HF_TOKEN")
    # Workspace/time no wandb e modo offline (nos sem internet -> `wandb sync` depois).
    [[ -n "${WANDB_ENTITY:-}" ]] && secrets+=(--env "WANDB_ENTITY=$WANDB_ENTITY")
    [[ -n "${WANDB_MODE:-}" ]] && secrets+=(--env "WANDB_MODE=$WANDB_MODE")
    apptainer exec --nv --cleanenv \
        --home "$CONTAINER_HOME" \
        -B "$REPO_DIR:/app" \
        -B "$RAID:$RAID" \
        --pwd /app \
        --env HF_HOME="$CACHE_DIR/hf" \
        --env HF_LEROBOT_HOME="$CACHE_DIR/hf/lerobot" \
        --env OPENPI_DATA_HOME="$CACHE_DIR/openpi" \
        --env JAX_COMPILATION_CACHE_DIR="$CACHE_DIR/jax" \
        --env WANDB_DIR="$CACHE_DIR/wandb" \
        --env WANDB_CACHE_DIR="$CACHE_DIR/wandb" \
        --env XDG_CACHE_HOME="$CACHE_DIR/xdg" \
        --env TMPDIR="$CACHE_DIR/tmp" \
        --env XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}" \
        "${secrets[@]}" \
        "$IMAGE" "$@"
}
