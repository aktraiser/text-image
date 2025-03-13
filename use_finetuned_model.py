import os
import sys
import torch
import logging
import argparse
from PIL import Image
import datetime
import subprocess
from diffusers import DiffusionPipeline, StableDiffusionPipeline

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Installe les dépendances nécessaires."""
    dependencies = [
        "diffusers", "transformers", "accelerate", "safetensors"
    ]
    
    for dep in dependencies:
        try:
            logger.info(f"Installation de {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except Exception as e:
            logger.warning(f"Erreur lors de l'installation de {dep}: {str(e)}")

def run_inference(prompt, model_dir="./hf_model_export", output_dir="./generated_outputs", num_steps=30, guidance_scale=7.5):
    """Exécute l'inférence avec le modèle fine-tuné."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        logger.info(f"Chargement du modèle depuis {model_dir}...")
        
        # Essayer de charger le modèle fine-tuné
        if os.path.exists(os.path.join(model_dir, "config.json")):
            # Utiliser le modèle de base Wan2.1 mais avec nos poids fine-tunés
            pipeline = StableDiffusionPipeline.from_pretrained(
                "Wan-AI/Wan2.1-T2V-14B",
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            
            # Charger les poids fine-tunés si disponibles
            if os.path.exists(os.path.join(model_dir, "unet")):
                logger.info("Chargement des poids UNet fine-tunés...")
                pipeline.unet.load_state_dict(torch.load(os.path.join(model_dir, "unet", "diffusion_pytorch_model.bin")))
            
            pipeline = pipeline.to("cuda")
            
            # Générer l'image
            logger.info(f"Génération d'image avec le prompt: {prompt}")
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt=prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            # Sauvegarder l'image générée
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:20])
            output_path = os.path.join(output_dir, f"{safe_prompt}_{timestamp}.png")
            
            image.save(output_path)
            logger.info(f"Image sauvegardée à: {output_path}")
            return output_path
        else:
            logger.error(f"Modèle non trouvé dans {model_dir}")
            return None
            
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description="Inférence avec le modèle fine-tuné")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt pour la génération d'image")
    parser.add_argument("--model_dir", type=str, default="./hf_model_export", help="Chemin vers le modèle fine-tuné")
    parser.add_argument("--output_dir", type=str, default="./generated_outputs", help="Dossier de sortie")
    parser.add_argument("--steps", type=int, default=30, help="Nombre d'étapes d'inférence")
    parser.add_argument("--guidance", type=float, default=7.5, help="Échelle de guidance")
    
    args = parser.parse_args()
    
    # Installer les dépendances
    install_dependencies()
    
    # Exécuter l'inférence
    output_path = run_inference(
        args.prompt,
        args.model_dir,
        args.output_dir,
        args.steps,
        args.guidance
    )
    
    if output_path:
        logger.info(f"Inférence terminée avec succès. Image sauvegardée à: {output_path}")
    else:
        logger.error("Échec de l'inférence")

if __name__ == "__main__":
    main() 
