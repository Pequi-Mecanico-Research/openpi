#!/bin/bash

echo "Iniciando cliente em modo evaluation (10 episódios) para testar as métricas do Eval_VLA e o fluxo de execução."

cd "$(dirname "$0")/examples/trossen_ai"

# Flags extras passadas ao script (ex.: ./client_inference_evaluation.sh --n_episodes 3 --max_steps 300)
# vêm por último e sobrescrevem as abaixo.
uv run main.py \
  --policy_host 192.168.0.62 \
  --policy_port 8000 \
  --mode evaluation \
  --n_episodes 10 \
  --max_steps 900 \
  --task_prompt "Organize the table" \
  "$@"
