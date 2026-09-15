#!/bin/bash

echo "Iniciando o Server de política do pi."

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_widowxai_organize_table \
  --policy.dir=path_to_police_localy