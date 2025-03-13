#!/bin/bash

# Vérifier si un prompt a été fourni
if [ "$#" -lt 1 ]; then
    echo "Usage: ./use_finetuned_model.sh \"votre prompt ici\" [model_dir] [base_model_dir] [output_dir]"
    exit 1
fi

PROMPT="$1"
MODEL_DIR="${2:-./hf_model_export}"
BASE_MODEL_DIR="${3:-./Wan2.1-T2V-14B}"
OUTPUT_DIR="${4:-./generated_outputs}"

# Installer les dépendances nécessaires
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install safetensors
pip install easydict einops decord opencv-python timm omegaconf imageio imageio-ffmpeg
pip install ftfy regex tqdm matplotlib scikit-image lpips kornia
pip install av
pip install dashscope aliyun-python-sdk-core

# Exécuter l'inférence avec le modèle fine-tuné
python use_finetuned_model.py --prompt "$PROMPT" --model_dir "$MODEL_DIR" --base_model_dir "$BASE_MODEL_DIR" --output_dir "$OUTPUT_DIR"

echo "Inférence terminée. Vérifiez le dossier $OUTPUT_DIR pour les résultats." 
