#!/bin/bash

# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create conda environment
conda create -n wmt2026 python=3.11 -y
source activate wmt2026

# Clone your private GitHub repo
git clone https://github.com/Benjamin-Pong/quality-estimation.git "/home/ubuntu/Machine Translation Metrics"

# Install local COMET
cd "/home/ubuntu/Machine Translation Metrics/COMET"
pip install -e .
pip install poetry
poetry install

# Install other dependencies
pip install python-dotenv

# Copy .env and jsonl files from local machine to instance
# Run these separately from your LOCAL WSL terminal after setup:
# scp "/mnt/c/Users/Benjamin Pong/OneDrive/Documents/Machine Translation Metrics/WMT2026/.env" scrawny-moccasin-jay:~/"Machine Translation Metrics/WMT2026/"
# scp "/mnt/c/Users/Benjamin Pong/OneDrive/Documents/Machine Translation Metrics/WMT2026/wmt25-genmt-humeval.jsonl" scrawny-moccasin-jay:~/"Machine Translation Metrics/WMT2026/"

echo "Setup complete!"