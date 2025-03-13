import os
import sys
import torch
import logging
import argparse
from PIL import Image
import datetime
import subprocess
import shutil

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Installe les dépendances nécessaires."""
    dependencies = [
        "diffusers", "transformers", "accelerate", "safetensors",
        "easydict", "einops", "decord", "opencv-python", "timm", 
        "omegaconf", "imageio", "imageio-ffmpeg", "ftfy", "regex", 
        "tqdm", "matplotlib", "scikit-image", "lpips", "kornia", "av",
        "dashscope", "aliyun-python-sdk-core"
    ]
    
    for dep in dependencies:
        try:
            logger.info(f"Installation de {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except Exception as e:
            logger.warning(f"Erreur lors de l'installation de {dep}: {str(e)}")

def run_inference(prompt, model_dir="./hf_model_export", base_model_dir="./Wan2.1-T2V-14B", 
                 output_dir="./generated_outputs", num_steps=30, guidance_scale=7.5):
    """Exécute l'inférence en utilisant directement le script generate.py du dépôt Wan2.1."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Cloner le dépôt si nécessaire
    if not os.path.exists("Wan2_1"):
        logger.info("Clonage du dépôt Wan2.1...")
        subprocess.run(["git", "clone", "https://github.com/Wan-Video/Wan2.1.git", "Wan2_1"], check=True)
    
    # Ajouter le dépôt au chemin Python
    sys.path.insert(0, os.path.abspath("Wan2_1"))
    
    try:
        # Créer un dossier temporaire pour combiner le modèle de base et les poids fine-tunés
        temp_model_dir = "./temp_model"
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)
        os.makedirs(temp_model_dir, exist_ok=True)
        
        # Copier les fichiers du modèle de base
        logger.info(f"Copie des fichiers du modèle de base depuis {base_model_dir}...")
        for item in os.listdir(base_model_dir):
            s = os.path.join(base_model_dir, item)
            d = os.path.join(temp_model_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        
        # Copier les fichiers du modèle fine-tuné s'ils existent
        logger.info(f"Copie des fichiers du modèle fine-tuné depuis {model_dir}...")
        if os.path.exists(model_dir):
            for item in os.listdir(model_dir):
                if item == "MODEL_WEIGHTS_NOT_INCLUDED.txt":
                    continue  # Ignorer ce fichier
                
                s = os.path.join(model_dir, item)
                d = os.path.join(temp_model_dir, item)
                
                if os.path.isdir(s):
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
        
        # Vérifier si nous avons des poids LoRA à appliquer
        lora_dir = os.path.join("outputs", "lora")
        if os.path.exists(lora_dir):
            logger.info(f"Copie des poids LoRA depuis {lora_dir}...")
            lora_dest = os.path.join(temp_model_dir, "lora")
            if os.path.exists(lora_dest):
                shutil.rmtree(lora_dest)
            shutil.copytree(lora_dir, lora_dest)
        
        # Importer le module generate
        try:
            logger.info("Importation du module generate...")
            from Wan2_1.generate import generate
            
            # Générer l'image
            logger.info(f"Génération d'image avec le prompt: {prompt}")
            logger.info(f"Utilisation du modèle dans: {temp_model_dir}")
            
            output = generate(
                temp_model_dir,
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            )
            
            # Sauvegarder l'image générée
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:20])
            output_path = os.path.join(output_dir, f"{safe_prompt}_{timestamp}.png")
            
            if isinstance(output, list) and len(output) > 0 and hasattr(output[0], "save"):
                output[0].save(output_path)
                logger.info(f"Image sauvegardée à: {output_path}")
                return output_path
            else:
                logger.error(f"Format de sortie non reconnu: {type(output)}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur lors de l'utilisation du module generate: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        # Nettoyer le dossier temporaire
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)

def main():
    parser = argparse.ArgumentParser(description="Inférence avec le modèle fine-tuné")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt pour la génération d'image")
    parser.add_argument("--model_dir", type=str, default="./hf_model_export", help="Chemin vers le modèle fine-tuné")
    parser.add_argument("--base_model_dir", type=str, default="./Wan2.1-T2V-14B", help="Chemin vers le modèle de base")
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
        args.base_model_dir,
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
