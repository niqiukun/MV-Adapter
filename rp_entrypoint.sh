#!/usr/bin/env bash

cd /runpod-volume/MV-Adapter || exit 1
git pull
source venv/bin/activate
apt update
apt install -y libegl1-mesa-dev libgl1-mesa-dev
TORCH_HOME=/runpod-volume/MV-Adapter/.cache/torch HF_HOME=./hf_cache exec python -u rp_handler.py
