import torch
from diffusers import DiffusionPipeline
import os
import logging
import datetime
import argparse
from PIL import Image
import sys
import importlib.util
from huggingface_hub import snapshot_download

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Installe les dépendances nécessaires pour le modèle Wan2.1."""
    import subprocess
    logger.info("Installation des dépendances pour Wan2.1...")
    
    dependencies = [
        "easydict",
        "einops",
        "decord",
        "opencv-python",
        "timm",
        "omegaconf",
        "imageio",
        "imageio-ffmpeg",
        "ftfy",
        "regex",
        "tqdm",
        "matplotlib",
        "scikit-image",
        "lpips",
        "kornia",
        "av"
    ]
    
    for dep in dependencies:
        try:
            logger.info(f"Installation de {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            logger.info(f"{dep} installé avec succès")
        except Exception as e:
            logger.warning(f"Erreur lors de l'installation de {dep}: {str(e)}")

def load_module_from_file(file_path, module_name):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        return None
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def clone_wan_repository():
    """Clone the Wan2.1 repository to access the model code."""
    logger.info("Cloning the Wan2.1 repository...")
    
    if not os.path.exists("Wan2_1"):  # Changed directory name to avoid the dot
        import subprocess
        subprocess.run(["git", "clone", "https://github.com/Wan-Video/Wan2.1.git", "Wan2_1"], check=True)
        logger.info("Repository cloned successfully.")
    else:
        logger.info("Repository already exists, skipping clone.")
    
    # Add the repository to Python path to import modules
    sys.path.append(os.path.abspath("Wan2_1"))

def load_model(model_dir="./Wan2.1-T2V-14B"):
    """Charge le modèle Wan2.1-T2V-14B."""
    logger.info(f"Chargement du modèle depuis {model_dir}...")
    
    # Installer les dépendances nécessaires
    install_dependencies()
    
    # Vérifier si le modèle existe localement
    if not os.path.exists(model_dir):
        logger.info(f"Modèle non trouvé localement, téléchargement depuis Hugging Face...")
        snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-14B", local_dir=model_dir)
    
    # Essayer de charger le modèle avec diffusers
    try:
        model = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        model = model.to("cuda")
        logger.info("Modèle chargé avec succès via DiffusionPipeline")
        return model
    except Exception as e:
        logger.warning(f"Erreur lors du chargement avec DiffusionPipeline: {str(e)}")
    
    # Méthode alternative: utiliser le code du dépôt Wan2.1
    try:
        clone_wan_repository()
        
        # Essayer d'importer le module de génération de Wan2.1
        try:
            # Importer directement depuis le dépôt cloné
            sys.path.insert(0, os.path.abspath("Wan2_1"))
            
            try:
                import easydict
                logger.info("Module easydict importé avec succès")
            except ImportError:
                logger.error("Module easydict non trouvé malgré l'installation")
            
            try:
                from Wan2_1.generate import generate
                logger.info("Module generate importé avec succès")
                
                # Créer une classe wrapper pour le modèle Wan
                class WanModelWrapper:
                    def __init__(self, model_path):
                        self.model_path = model_path
                    
                    def generate(self, prompt, **kwargs):
                        return generate(
                            self.model_path, 
                            prompt=prompt,
                            **kwargs
                        )
                
                model = WanModelWrapper(model_dir)
                logger.info("Modèle Wan2.1 chargé avec succès via le module generate")
                return model
                
            except ImportError as e:
                logger.error(f"Erreur lors de l'importation du module generate: {str(e)}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'importation du module Wan2.1: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
    
    logger.error("Impossible de charger le modèle par aucune méthode")
    return None

def run_inference(model, prompt, output_dir="./generated_outputs", num_inference_steps=30, guidance_scale=7.5):
    """Exécute l'inférence avec le modèle."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Exécution de l'inférence avec le prompt: {prompt}")
    
    try:
        # Générer l'image
        if hasattr(model, "generate"):
            # Pour le wrapper Wan
            output = model.generate(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        else:
            # Pour le pipeline de diffusion standard
            output = model(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        # Sauvegarder l'image générée
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:20])
        output_path = os.path.join(output_dir, f"{safe_prompt}_{timestamp}.png")
        
        if isinstance(output, list) and hasattr(output[0], "save"):
            output[0].save(output_path)
        elif hasattr(output, "save"):
            output.save(output_path)
        else:
            logger.warning("Format de sortie non reconnu, impossible de sauvegarder l'image")
            return None
        
        logger.info(f"Image générée sauvegardée à: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description="Inférence avec le modèle Wan2.1-T2V-14B")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt pour la génération d'image")
    parser.add_argument("--model_dir", type=str, default="./Wan2.1-T2V-14B", help="Chemin vers le modèle")
    parser.add_argument("--output_dir", type=str, default="./generated_outputs", help="Dossier de sortie")
    parser.add_argument("--steps", type=int, default=30, help="Nombre d'étapes d'inférence")
    parser.add_argument("--guidance", type=float, default=7.5, help="Échelle de guidance")
    
    args = parser.parse_args()
    
    # Charger le modèle
    model = load_model(args.model_dir)
    
    if model is None:
        logger.error("Échec du chargement du modèle, arrêt du programme")
        return
    
    # Exécuter l'inférence
    output_path = run_inference(
        model, 
        args.prompt, 
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
