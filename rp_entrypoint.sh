#!/usr/bin/env bash

cd /runpod-volume/MV-Adapter || exit 1
git pull
TORCH_HOME=/runpod-volume/MV-Adapter/.cache/torch HF_HOME=./hf_cache exec /runpod-volume/MV-Adapter/venv/bin/python -u rp_handler.py
