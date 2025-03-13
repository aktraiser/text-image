import torch
from diffusers import DiffusionPipeline
import os
import logging
import datetime
import argparse
from PIL import Image
import sys
import importlib.util
import subprocess
import glob
import json

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

def is_full_model(model_dir):
    """Détecte si le dossier contient un modèle complet (fusionné) ou un modèle de base."""
    # Vérifier la présence du fichier de configuration d'inférence
    if os.path.exists(os.path.join(model_dir, "inference_config.json")):
        return True
    
    # Vérifier la présence du dossier lora
    if os.path.exists(os.path.join(model_dir, "lora")):
        return True
    
    return False

def load_model(model_dir="/workspace/full_model"):
    """Charge le modèle Wan2.1-T2V-14B depuis le dossier spécifié."""
    logger.info(f"Chargement du modèle depuis {model_dir}...")
    
    # Installer les dépendances nécessaires
    install_dependencies()
    
    # Vérifier si le modèle existe
    if not os.path.exists(model_dir):
        logger.error(f"Modèle non trouvé dans {model_dir}")
        
        # Vérifier si le modèle existe dans le dossier par défaut
        default_model_dir = "./Wan2.1-T2V-14B"
        if os.path.exists(default_model_dir):
            logger.info(f"Utilisation du modèle de base dans {default_model_dir}")
            model_dir = default_model_dir
        else:
            logger.error("Modèle non trouvé localement")
            return None
    
    # Vérifier si c'est un modèle complet ou un modèle de base
    is_merged_model = is_full_model(model_dir)
    if is_merged_model:
        logger.info("Modèle complet détecté")
    else:
        logger.info("Modèle de base détecté")
    
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
        logger.warning(f"Erreur lors du chargement du modèle avec DiffusionPipeline: {str(e)}")
    
    # Si le chargement avec diffusers échoue, essayer d'utiliser le code de Wan2.1
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
            
            # Utiliser le script generate.py directement via subprocess
            logger.info("Utilisation du script generate.py via subprocess")
            
            # Créer une classe wrapper pour le modèle Wan
            class WanModelWrapper:
                def __init__(self, model_path):
                    self.model_path = model_path
                
                def __call__(self, prompt, num_inference_steps=30, guidance_scale=7.5, size="832*480"):
                    output_dir = "./generated_outputs"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    cmd = [
                        sys.executable,
                        "Wan2_1/generate.py",
                        "--task", "t2v-14B",
                        "--size", size,
                        "--ckpt_dir", self.model_path,
                        "--prompt", prompt,
                        "--output_dir", output_dir,
                        "--num_inference_steps", str(num_inference_steps),
                        "--guidance_scale", str(guidance_scale)
                    ]
                    
                    # Si c'est un modèle complet avec dossier lora, ajouter le chemin du lora
                    lora_dir = os.path.join(self.model_path, "lora")
                    if os.path.exists(lora_dir):
                        lora_files = glob.glob(os.path.join(lora_dir, "*.safetensors"))
                        if lora_files:
                            lora_file = lora_files[0]
                            cmd.extend(["--lora_path", lora_file, "--lora_scale", "0.8"])
                    
                    logger.info(f"Exécution de la commande: {' '.join(cmd)}")
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    
                    # Chercher le fichier généré le plus récent
                    files = glob.glob(f"{output_dir}/*.mp4") + glob.glob(f"{output_dir}/*.gif") + glob.glob(f"{output_dir}/*.png")
                    if files:
                        latest_file = max(files, key=os.path.getctime)
                        logger.info(f"Fichier généré: {latest_file}")
                        
                        # Créer un objet de type "images" pour être compatible avec l'API de diffusers
                        if latest_file.endswith(('.png', '.jpg', '.jpeg')):
                            image = Image.open(latest_file)
                            class ImageContainer:
                                def __init__(self, images):
                                    self.images = images
                            return ImageContainer([image])
                        else:
                            logger.info(f"Fichier vidéo généré: {latest_file}")
                            class VideoContainer:
                                def __init__(self, video_path):
                                    self.video_path = video_path
                                    self.images = [Image.open(latest_file.replace('.mp4', '.png')) if os.path.exists(latest_file.replace('.mp4', '.png')) else None]
                            return VideoContainer(latest_file)
                    else:
                        logger.error("Aucun fichier généré")
                        return None
            
            model = WanModelWrapper(model_dir)
            logger.info("Modèle Wan2.1 chargé avec succès via le wrapper")
            return model
                
        except ImportError as e:
            logger.error(f"Erreur lors de l'importation du module generate: {str(e)}")
                
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
    
    logger.error("Impossible de charger le modèle par aucune méthode")
    return None

def run_inference(model, prompt, output_dir="./generated_outputs", num_inference_steps=30, guidance_scale=7.5, size="832*480"):
    """Exécute l'inférence avec le modèle."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Exécution de l'inférence avec le prompt: {prompt}")
    
    try:
        # Générer l'image ou la vidéo
        if hasattr(model, "__call__"):
            # Pour le wrapper Wan
            output = model(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                size=size
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
        
        if hasattr(output, "images") and output.images and output.images[0]:
            output.images[0].save(output_path)
            logger.info(f"Image générée sauvegardée à: {output_path}")
            return output_path
        elif isinstance(output, list) and hasattr(output[0], "save"):
            output[0].save(output_path)
            logger.info(f"Image générée sauvegardée à: {output_path}")
            return output_path
        elif hasattr(output, "save"):
            output.save(output_path)
            logger.info(f"Image générée sauvegardée à: {output_path}")
            return output_path
        elif hasattr(output, "video_path"):
            logger.info(f"Vidéo générée sauvegardée à: {output.video_path}")
            return output.video_path
        else:
            logger.warning(f"Format de sortie non reconnu: {type(output)}")
            return None
    
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description="Inférence avec le modèle Wan2.1-T2V-14B")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt pour la génération")
    parser.add_argument("--model_dir", type=str, default="/workspace/full_model", help="Chemin vers le modèle")
    parser.add_argument("--output_dir", type=str, default="./generated_outputs", help="Dossier de sortie")
    parser.add_argument("--steps", type=int, default=30, help="Nombre d'étapes d'inférence")
    parser.add_argument("--guidance", type=float, default=7.5, help="Échelle de guidance")
    parser.add_argument("--size", type=str, default="832*480", help="Taille de la vidéo (832*480 ou 1280*720)")
    
    args = parser.parse_args()
    
    # Charger le modèle
    model = load_model(args.model_dir)
    
    if model:
        # Exécuter l'inférence
        output_path = run_inference(
            model, 
            args.prompt, 
            args.output_dir, 
            args.steps, 
            args.guidance,
            args.size
        )
        
        if output_path:
            logger.info(f"Inférence terminée avec succès. Fichier sauvegardé à: {output_path}")
        else:
            logger.error("Échec de l'inférence")
    else:
        logger.error("Impossible de charger le modèle")

if __name__ == "__main__":
    main() 
