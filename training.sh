#!/bin/bash

# Vérifier si on est en mode entraînement ou inférence
MODE="train"
PROMPT=""
OUTPUT_DIR="./generated_outputs"
SIZE="832*480"

# Traiter les arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --inference)
      MODE="inference"
      shift
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --size)
      SIZE="$2"
      shift 2
      ;;
    *)
      echo "Option inconnue: $1"
      exit 1
      ;;
  esac
done

# Fonction pour installer les dépendances
install_dependencies() {
  echo "Installation des dépendances..."
  pip install --upgrade pip
  pip install torch torchvision torchaudio
  pip install diffusers transformers accelerate peft
  pip install bitsandbytes
  pip install wandb
  pip install safetensors
  pip install ftfy
  pip install trl
  pip install easydict einops decord opencv-python timm omegaconf imageio imageio-ffmpeg
  pip install regex tqdm matplotlib scikit-image lpips kornia
  pip install av
}

# Mode entraînement
if [ "$MODE" = "train" ]; then
  echo "=== Mode entraînement ==="
  install_dependencies
  
  # Lancer l'entraînement
  echo "Lancement de l'entraînement..."
  python training.py
  
  echo "Entraînement terminé. Les poids LoRA sont disponibles dans le dossier hf_model_export."
  echo "Pour générer des images, utilisez: ./training.sh --inference --prompt \"votre prompt ici\""

# Mode inférence
else
  echo "=== Mode inférence ==="
  
  # Vérifier si un prompt a été fourni
  if [ -z "$PROMPT" ]; then
    echo "ERREUR: Aucun prompt fourni pour l'inférence."
    echo "Usage: ./training.sh --inference --prompt \"votre prompt ici\" [--output_dir dossier] [--size largeur*hauteur]"
    exit 1
  fi
  
  # Installer les dépendances nécessaires pour l'inférence
  install_dependencies
  
  # Créer le répertoire de sortie s'il n'existe pas
  mkdir -p "$OUTPUT_DIR"
  
  echo "=== Conversion des poids LoRA au format diffusers ==="
  
  # Vérifier si le dossier hf_model_export existe
  if [ ! -d "./hf_model_export" ]; then
    echo "ERREUR: Le dossier hf_model_export n'existe pas."
    echo "Veuillez exécuter le script en mode entraînement d'abord: ./training.sh"
    exit 1
  fi
  
  # Vérifier si le dossier diffusers_format existe déjà
  if [ ! -d "./hf_model_export/diffusers_format" ]; then
    echo "Conversion des poids LoRA au format diffusers..."
    
    # Exécuter le script de conversion intégré dans training.py
    python -c "
import sys
sys.path.append('.')
from training import export_model
export_model(None, None, 'hf_model_export')
"
    
    if [ $? -ne 0 ]; then
      echo "ERREUR: La conversion des poids LoRA a échoué."
      exit 1
    fi
    
    echo "Conversion terminée avec succès."
  else
    echo "Les poids LoRA sont déjà au format diffusers."
  fi
  
  echo "=== Lancement de l'inférence ==="
  
  # Exécuter l'inférence avec les poids LoRA convertis
  echo "Génération d'image avec le prompt: $PROMPT"
  python inference.py \
    --prompt "$PROMPT" \
    --model_path "./hf_model_export/diffusers_format" \
    --output_dir "$OUTPUT_DIR" \
    --size "$SIZE" \
    --num_inference_steps 50 \
    --guidance_scale 7.5
  
  echo "Inférence terminée. Vérifiez le dossier $OUTPUT_DIR pour les résultats."
fi
