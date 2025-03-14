#!/bin/bash

# Vérifier si un prompt a été fourni
if [ "$#" -lt 1 ]; then
    echo "Usage: ./inference.sh \"votre prompt ici\" [model_path] [output_dir] [size] [use_lora_only]"
    echo "  - model_path: Chemin vers le modèle (défaut: /workspace/full_model)"
    echo "  - output_dir: Dossier de sortie (défaut: ./generated_outputs)"
    echo "  - size: Taille de l'image (défaut: 832*480)"
    echo "  - use_lora_only: Utiliser uniquement les poids LoRA (true/false, défaut: false)"
    exit 1
fi

PROMPT="$1"
MODEL_PATH="${2:-/workspace/full_model}"
OUTPUT_DIR="${3:-./generated_outputs}"
SIZE="${4:-832*480}"
USE_LORA_ONLY="${5:-false}"

# Installer les dépendances nécessaires
echo "Installation des dépendances..."
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install safetensors
pip install easydict einops decord opencv-python timm omegaconf imageio imageio-ffmpeg
pip install ftfy regex tqdm matplotlib scikit-image lpips kornia
pip install av

# Créer le répertoire de sortie s'il n'existe pas
mkdir -p "$OUTPUT_DIR"

# Préparer les arguments pour inference.py
ARGS=""

# Ajouter le prompt
ARGS="$ARGS --prompt \"$PROMPT\""

# Ajouter le chemin du modèle
if [ "$USE_LORA_ONLY" = "true" ]; then
    echo "Utilisation des poids LoRA uniquement..."
    ARGS="$ARGS --use_lora_only"
    # Si le chemin du modèle n'est pas spécifié, utiliser hf_model_export
    if [ "$2" = "" ]; then
        MODEL_PATH="./hf_model_export"
    fi
fi

ARGS="$ARGS --model_path \"$MODEL_PATH\""

# Ajouter le dossier de sortie et la taille
ARGS="$ARGS --output_dir \"$OUTPUT_DIR\" --size \"$SIZE\""

# Ajouter des paramètres supplémentaires pour les modèles de diffusion
ARGS="$ARGS --num_inference_steps 50 --guidance_scale 7.5"

# Vérifier que le chemin du modèle existe
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERREUR: Le chemin du modèle '$MODEL_PATH' n'existe pas."
    echo "Veuillez spécifier un chemin valide ou créer le répertoire."
    exit 1
fi

# Afficher les paramètres pour le débogage
echo "Paramètres d'inférence:"
echo "- Prompt: $PROMPT"
echo "- Chemin du modèle: $MODEL_PATH"
echo "- Dossier de sortie: $OUTPUT_DIR"
echo "- Taille: $SIZE"
echo "- Utiliser LoRA uniquement: $USE_LORA_ONLY"

# Exécuter l'inférence
echo "Lancement de l'inférence..."
eval "python inference.py $ARGS"

echo "Inférence terminée. Vérifiez le dossier $OUTPUT_DIR pour les résultats." 
