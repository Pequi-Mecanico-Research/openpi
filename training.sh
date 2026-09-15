#!/bin/bash

echo "Computing Norm Stats"

uv run scripts/compute_norm_stats.py \
    --config-name pi05_widowxai_organize_table

echo "Finished computing norm stats. Start Training..."

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_widowxai_organize_table \
    --exp-name=pi05_organize_cube_with_effort \
    --overwrite