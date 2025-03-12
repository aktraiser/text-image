#!/bin/bash

# Met à jour pip
pip install --upgrade pip

# Installe les dépendances nécessaires pour l'entraînement local de Wan2.1-T2V-14B
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate peft
pip install bitsandbytes  # Pour l'optimiseur adamw8bit
pip install wandb  # Pour le logging avec Weights & Biases
pip install safetensors  # Pour sauvegarder les poids efficacement

# Exécute le script Python
python training.py
