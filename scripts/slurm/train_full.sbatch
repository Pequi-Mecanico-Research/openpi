#!/bin/bash
# Fine-tuning FULL do pi0.5 (config pi05_widowxai_organize_table_full) em 1x H100, via Apptainer.
# Submeter da RAIZ do repo:   sbatch scripts/slurm/train_full.sbatch
# Retomar apos queda/manutencao/time limit: submeter o MESMO comando de novo (o job e reentrante).
# Overrides:   sbatch --export=ALL,EXP_NAME=bs32_v2,AUTO_REQUEUE=1,WANDB_ENTITY=<time> scripts/slurm/train_full.sbatch
#
#SBATCH --job-name=pi05-widowx-full
#SBATCH --partition=h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# Ajuste ao MaxTime da particao (`sinfo -o "%P %l"`). O sinal abaixo so dispara se houver time limit.
#SBATCH --time=48:00:00
# 300 s antes do fim: SIGUSR1 para o shell do batch -> repassado ao treino, que salva checkpoint e sai.
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/raid/%u/vla/logs/%x_%j.log

set -uo pipefail
source "${REPO_DIR:-$SLURM_SUBMIT_DIR}/scripts/slurm/env.sh"

EXP_NAME=${EXP_NAME:-full_h100_bs32}
# 1 = ao ser interrompido por sinal (com checkpoint salvo), recoloca o job na fila automaticamente.
AUTO_REQUEUE=${AUTO_REQUEUE:-0}
PREEMPTED_EXIT_CODE=99  # igual a scripts/train.py

if ! find "$REPO_DIR/assets/$CONFIG_NAME" -name norm_stats.json 2>/dev/null | grep -q .; then
    echo "ERRO: norm stats ausentes em $REPO_DIR/assets/$CONFIG_NAME. Rode scripts/slurm/norm_stats.sbatch antes." >&2
    exit 1
fi

# wandb e obrigatorio por padrao: sem credencial o job falha agora, em vez de treinar dias sem log.
# Para rodar sem wandb de proposito: --export=ALL,WANDB_ENABLED=0
WANDB_ENABLED=${WANDB_ENABLED:-1}
WANDB_FLAG=()
if [[ "$WANDB_ENABLED" == "0" ]]; then
    echo "wandb desabilitado (WANDB_ENABLED=0)."
    WANDB_FLAG=(--no-wandb-enabled)
elif [[ "${WANDB_MODE:-online}" != "offline" && -z "${WANDB_API_KEY:-}" && ! -f "$CONTAINER_HOME/.netrc" ]]; then
    echo "ERRO: wandb habilitado, mas sem WANDB_API_KEY nem $CONTAINER_HOME/.netrc." >&2
    echo "      Configure a credencial (docs/full_finetune_h100_slurm.md, secao 6.4) ou submeta com WANDB_ENABLED=0." >&2
    exit 1
else
    echo "wandb: modo ${WANDB_MODE:-online}, entity ${WANDB_ENTITY:-<default da conta>}"
fi

echo "Job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) | node $(hostname) | config $CONFIG_NAME | exp $EXP_NAME"
echo "Checkpoints: $CKPT_BASE/$CONFIG_NAME/$EXP_NAME"

# --resume sempre: se o diretorio nao existe ou nao tem checkpoint, o openpi comeca do zero;
# se existe, continua do ultimo passo salvo. Nunca usamos --overwrite aqui (apagaria checkpoints).
run_in_container python scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --checkpoint-base-dir="$CKPT_BASE" \
    --resume \
    "${WANDB_FLAG[@]}" \
    "$@" &
CHILD=$!

forward() {
    echo "$(date +%T) Recebido $1 -> repassando ao treino (PID $CHILD) para salvar checkpoint."
    kill -s "$1" "$CHILD" 2>/dev/null
}
trap 'forward USR1' USR1
trap 'forward TERM' TERM

# `wait` retorna cedo quando um trap dispara; espera ate o processo realmente terminar.
wait "$CHILD"
RC=$?
while kill -0 "$CHILD" 2>/dev/null; do
    wait "$CHILD"
    RC=$?
done

echo "$(date +%T) Treino terminou com codigo $RC"
if [[ $RC -eq $PREEMPTED_EXIT_CODE ]]; then
    if [[ "$AUTO_REQUEUE" == "1" ]]; then
        echo "Checkpoint salvo; recolocando o job na fila (scontrol requeue)."
        scontrol requeue "$SLURM_JOB_ID"
    else
        echo "Checkpoint salvo. Para continuar: sbatch scripts/slurm/train_full.sbatch (mesmo EXP_NAME)."
    fi
fi
exit $RC
