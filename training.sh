#!/bin/bash
# Script de lancement pour l'installation des librairies et le finetuning

# Quitter immédiatement en cas d'erreur
set -e
set -o pipefail

echo "=== Mise à jour de pip et installation des librairies requises ==="

# Mettre à jour pip
pip install --upgrade pip

# Installation des packages nécessaires
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate peft
pip install bitsandbytes
pip install wandb
pip install safetensors
pip install ftfy
pip install trl

echo "=== Lancement du script de finetuning ==="
python training.py

echo "=== Fin de l'entraînement ==="
