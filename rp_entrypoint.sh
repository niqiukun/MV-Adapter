#!/usr/bin/env bash

cd /runpod-volume/MV-Adapter || exit 1
git pull
apt update
apt install -y libegl1-mesa-dev libgl1-mesa-dev
PATH=/runpod-volume/MV-Adapter/venv/bin:$PATH TORCH_HOME=/runpod-volume/MV-Adapter/.cache/torch HF_HOME=./hf_cache exec /runpod-volume/MV-Adapter/venv/bin/python -u rp_handler.py
