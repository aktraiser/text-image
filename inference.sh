#!/bin/bash

# Vérifier si un prompt a été fourni
if [ "$#" -lt 1 ]; then
    echo "Usage: ./inference.sh \"votre prompt ici\" [model_path] [output_dir] [size]"
    exit 1
fi

PROMPT="$1"
MODEL_PATH="${2:-/workspace/full_model}"
OUTPUT_DIR="${3:-./generated_outputs}"
SIZE="${4:-832*480}"

# Installer les dépendances nécessaires
echo "Installation des dépendances..."
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install safetensors
pip install easydict einops decord opencv-python timm omegaconf imageio imageio-ffmpeg
pip install ftfy regex tqdm matplotlib scikit-image lpips kornia
pip install av

# Exécuter l'inférence
echo "Lancement de l'inférence avec le prompt: $PROMPT"
python inference.py --prompt "$PROMPT" --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR" --size "$SIZE"

echo "Inférence terminée. Vérifiez le dossier $OUTPUT_DIR pour les résultats." 
