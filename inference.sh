#!/bin/bash

# Vérifier si un prompt a été fourni
if [ "$#" -lt 1 ]; then
    echo "Usage: ./inference.sh \"votre prompt ici\" [model_dir] [output_dir]"
    exit 1
fi

PROMPT="$1"
MODEL_DIR="${2:-./Wan2.1-T2V-14B}"
OUTPUT_DIR="${3:-./generated_outputs}"

# Installer les dépendances nécessaires
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install safetensors
pip install easydict einops decord opencv-python timm omegaconf imageio imageio-ffmpeg

# Exécuter l'inférence
python inference.py --prompt "$PROMPT" --model_dir "$MODEL_DIR" --output_dir "$OUTPUT_DIR"

echo "Inférence terminée. Vérifiez le dossier $OUTPUT_DIR pour les résultats." 
