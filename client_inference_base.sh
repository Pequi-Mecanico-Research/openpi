#!/bin/bash

echo "Iniciando cliente para inferência do pi05 (braço + base móvel, ROS 2 Jazzy no Docker)."

cd "$(dirname "$0")/examples/trossen_ai"

# --build reconstrói a imagem se o código mudou (usa cache, é rápido quando nada mudou).
# Flags extras passadas ao script ao rodar o script (ex.: ./client_inference_base.sh --mode test) sobrescrevem as abaixo.
docker compose run --rm --build trossen_openpi_base \
  python main.py \
  --policy_host 192.168.0.62 \
  --policy_port 8000 \
  --mode autonomous_base \
  --base_topic /cmd_vel \
  --max_steps 5000 \
  --task_prompt "Organize the table" \
  "$@"
