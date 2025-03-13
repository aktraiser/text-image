#!/bin/bash

# Vérifier si un prompt a été fourni
if [ "$#" -lt 1 ]; then
    echo "Usage: ./direct_inference.sh \"votre prompt ici\" [model_dir] [base_model_dir] [output_dir] [trigger_word]"
    exit 1
fi

PROMPT="$1"
MODEL_DIR="${2:-./hf_model_export}"
BASE_MODEL_DIR="${3:-./Wan2.1-T2V-14B}"
OUTPUT_DIR="${4:-./generated_outputs}"
TRIGGER_WORD="${5:-}"
SIZE="${6:-832*480}"

# Installer les dépendances nécessaires
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install safetensors
pip install easydict einops decord opencv-python timm omegaconf imageio imageio-ffmpeg
pip install ftfy regex tqdm matplotlib scikit-image lpips kornia
pip install av

# Exécuter l'inférence directe
if [ -z "$TRIGGER_WORD" ]; then
    python direct_inference.py --prompt "$PROMPT" --model_dir "$MODEL_DIR" --base_model_dir "$BASE_MODEL_DIR" --output_dir "$OUTPUT_DIR" --size "$SIZE"
else
    python direct_inference.py --prompt "$PROMPT" --model_dir "$MODEL_DIR" --base_model_dir "$BASE_MODEL_DIR" --output_dir "$OUTPUT_DIR" --trigger_word "$TRIGGER_WORD" --size "$SIZE"
fi

echo "Inférence terminée. Vérifiez le dossier $OUTPUT_DIR pour les résultats." 
