#!/bin/bash

# Install required packages
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate peft
pip install bitsandbytes
pip install wandb
pip install safetensors
pip install ftfy
pip install trl

# Run the training script
python training.py
