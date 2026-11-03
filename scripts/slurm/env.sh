# Configuracao comum dos jobs Slurm do openpi. Carregado via `source` pelos .sbatch.
# Regra do cluster: nada de dados/caches na home -> tudo abaixo fica em /raid/$USER.
# Qualquer variavel pode ser sobrescrita no sbatch: `sbatch --export=ALL,EXP_NAME=teste scripts/slurm/train_full.sbatch`.

RAID=${RAID:-/raid/$USER}
# Pasta do projeto (teleop) no /raid: containers/, checkpoints/ e vla/ (logs, caches, repo) ficam juntos aqui.
TELEOP_DIR=${TELEOP_DIR:-$RAID/teleop}
PROJECT_DIR=${PROJECT_DIR:-$TELEOP_DIR/vla}
# Os .sbatch devem ser submetidos da raiz do repo (clonado dentro do /raid): o Slurm copia o script para um
# diretorio de spool, entao o caminho do repo vem do diretorio de submissao.
REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
IMAGE=${IMAGE:-$TELEOP_DIR/containers/openpi.sif}
CKPT_BASE=${CKPT_BASE:-$TELEOP_DIR/checkpoints/openpi}
CACHE_DIR=${CACHE_DIR:-$PROJECT_DIR/cache}
# Home "falsa" do container: ~/.cache, ~/.netrc (login do wandb), ~/.config, etc. ficam no /raid.
CONTAINER_HOME=${CONTAINER_HOME:-$PROJECT_DIR/home}

# Segredos (WANDB_API_KEY, HF_TOKEN, WANDB_ENTITY...) num arquivo fora do repo, para nao ir parar no git.
# Formato: uma linha CHAVE=valor por variavel. Mantenha com `chmod 600`.
ENV_FILE=${ENV_FILE:-$PROJECT_DIR/.env}
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

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
    # Segredos via APPTAINERENV_* (passam mesmo com --cleanenv): como `--env CHAVE=valor` eles apareceriam na linha
    # de comando do apptainer, visivel para qualquer usuario do no via `ps`.
    [[ -n "${WANDB_API_KEY:-}" ]] && export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
    [[ -n "${HF_TOKEN:-}" ]] && export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
    # Workspace/time no wandb e modo offline (nos sem internet -> `wandb sync` depois).
    [[ -n "${WANDB_ENTITY:-}" ]] && secrets+=(--env "WANDB_ENTITY=$WANDB_ENTITY")
    [[ -n "${WANDB_MODE:-}" ]] && secrets+=(--env "WANDB_MODE=$WANDB_MODE")
    # GPUs: este cluster NAO isola GPUs por cgroup, entao o job enxerga as 8 e so o CUDA_VISIBLE_DEVICES do Slurm
    # limita o uso. O --cleanenv apagaria essa variavel e o JAX pegaria todas as GPUs do no (inclusive as de outros
    # usuarios). Sem GPU alocada (ex.: norm_stats), esconde todas e forca o JAX em CPU.
    local gpu_env=(--env CUDA_DEVICE_ORDER=PCI_BUS_ID)
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        gpu_env+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
    else
        gpu_env+=(--env CUDA_VISIBLE_DEVICES= --env JAX_PLATFORMS=cpu)
    fi
    apptainer exec --nv --cleanenv \
        "${gpu_env[@]}" \
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
