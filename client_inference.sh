#!/bin/bash

echo "Iniciando cliente para inferência do pi05."

cd "$(dirname "$0")/examples/trossen_ai"

uv run main.py \
  --policy_host 192.168.0.62 \
  --policy_port 8000 \
  --mode autonomous \
  --max_steps 5000 \
  --task_prompt "Organize the table"