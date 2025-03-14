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
  
  echo "=== Vérification des poids LoRA ==="
  
  # Vérifier si le dossier hf_model_export existe
  if [ ! -d "./hf_model_export" ]; then
    echo "ERREUR: Le dossier hf_model_export n'existe pas."
    echo "Veuillez exécuter le script en mode entraînement d'abord: ./training.sh"
    exit 1
  fi
  
  # Vérifier si les dossiers LoRA existent
  if [ ! -d "./hf_model_export/unet_lora" ] || [ ! -d "./hf_model_export/text_encoder_lora" ]; then
    echo "ERREUR: Les dossiers de poids LoRA sont manquants."
    echo "Veuillez exécuter le script en mode entraînement d'abord: ./training.sh"
    exit 1
  fi
  
  echo "=== Lancement de l'inférence avec LoRA ==="
  
  # Créer un script Python temporaire pour l'inférence
  cat > temp_inference.py << 'EOF'
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
import argparse
import os
import time

def parse_size(size_str):
    try:
        width, height = map(int, size_str.split('*'))
        return width, height
    except:
        print(f"Format de taille invalide: {size_str}, utilisation de la taille par défaut 512x512")
        return 512, 512

def main():
    parser = argparse.ArgumentParser(description="Inférence avec LoRA pour Stable Diffusion")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt pour la génération d'image")
    parser.add_argument("--output_dir", type=str, default="./generated_outputs", help="Dossier de sortie")
    parser.add_argument("--size", type=str, default="512*512", help="Taille de l'image (largeur*hauteur)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Nombre d'étapes d'inférence")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Échelle de guidance")
    
    args = parser.parse_args()
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Déterminer le type de données à utiliser
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere+ (RTX 30xx, A100, etc.)
            dtype = torch.bfloat16
            print("Utilisation de bfloat16 sur GPU Ampere+")
        else:  # Architectures plus anciennes
            dtype = torch.float16
            print("Utilisation de float16 sur GPU pré-Ampere")
    else:
        dtype = torch.float32
        print("Utilisation de float32 sur CPU")
    
    # Charger le modèle de base - Utiliser Stable Diffusion 2.1 comme dans l'entraînement
    print("Chargement du modèle de base Stable Diffusion 2.1...")
    base_model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if torch.cuda.is_available() else None,
    )
    
    # Déplacer le modèle sur GPU si disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Appliquer les adaptateurs LoRA avec PEFT
    print("Application des adaptateurs LoRA avec PEFT...")
    
    # Appliquer l'adaptateur LoRA à l'UNet
    if os.path.exists("./hf_model_export/unet_lora"):
        print("Application de l'adaptateur LoRA à l'UNet...")
        pipe.unet = PeftModel.from_pretrained(
            pipe.unet,
            "./hf_model_export/unet_lora",
            adapter_name="default"
        )
    
    # Appliquer l'adaptateur LoRA à l'encodeur de texte
    if os.path.exists("./hf_model_export/text_encoder_lora"):
        print("Application de l'adaptateur LoRA à l'encodeur de texte...")
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder,
            "./hf_model_export/text_encoder_lora",
            adapter_name="default"
        )
    
    # Traiter la taille
    width, height = parse_size(args.size)
    print(f"Taille d'image: {width}x{height}")
    
    # Générer l'image
    print(f"Génération d'image avec le prompt: {args.prompt}")
    start_time = time.time()
    
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        width=width,
        height=height
    ).images[0]
    
    # Sauvegarder l'image
    timestamp = int(time.time())
    output_file = os.path.join(args.output_dir, f"image_{timestamp}.png")
    image.save(output_file)
    
    print(f"Image générée et sauvegardée dans {output_file}")
    print(f"Temps de génération: {time.time() - start_time:.2f} secondes")

if __name__ == "__main__":
    main()
EOF
  
  # Exécuter le script d'inférence
  echo "Génération d'image avec le prompt: $PROMPT"
  python temp_inference.py \
    --prompt "$PROMPT" \
    --output_dir "$OUTPUT_DIR" \
    --size "$SIZE" \
    --num_inference_steps 50 \
    --guidance_scale 7.5
  
  # Supprimer le script temporaire
  rm temp_inference.py
  
  echo "Inférence terminée. Vérifiez le dossier $OUTPUT_DIR pour les résultats."
fi
