#!/bin/bash
# Script de lancement pour l'installation des librairies et l'entraînement

set -e
set -o pipefail

echo "=== Mise à jour de pip et installation des librairies requises ==="
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate peft
pip install bitsandbytes
pip install wandb
pip install safetensors
pip install ftfy
pip install trl

echo "=== Lancement du script de finetuning (training.py) ==="
python training.py

echo "=== Fin de l'entraînement ==="
