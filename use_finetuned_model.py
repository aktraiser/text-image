import os
import sys
import torch
import logging
import argparse
from PIL import Image
import datetime
import subprocess
import shutil
import glob

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

def find_lora_files():
    """Recherche les fichiers LoRA dans les dossiers courants."""
    lora_files = []
    
    # Rechercher dans le dossier outputs/lora
    if os.path.exists("outputs/lora"):
        lora_files.extend(glob.glob("outputs/lora/**/*.safetensors", recursive=True))
    
    # Rechercher dans le dossier hf_model_export
    if os.path.exists("hf_model_export"):
        lora_files.extend(glob.glob("hf_model_export/**/*.safetensors", recursive=True))
    
    # Rechercher dans le dossier courant
    lora_files.extend(glob.glob("./**/*.safetensors", recursive=True))
    
    return lora_files

def run_inference(prompt, model_dir="./hf_model_export", base_model_dir="./Wan2.1-T2V-14B", 
                 output_dir="./generated_outputs", num_steps=30, guidance_scale=7.5,
                 lora_weight=0.8, trigger_word=None):
    """Exécute l'inférence en utilisant directement le script generate.py du dépôt Wan2.1."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Rechercher les fichiers LoRA
    lora_files = find_lora_files()
    if lora_files:
        logger.info(f"Fichiers LoRA trouvés: {lora_files}")
    else:
        logger.warning("Aucun fichier LoRA trouvé")
    
    # Cloner le dépôt si nécessaire
    if not os.path.exists("Wan2_1"):
        logger.info("Clonage du dépôt Wan2.1...")
        subprocess.run(["git", "clone", "https://github.com/Wan-Video/Wan2.1.git", "Wan2_1"], check=True)
    
    # Ajouter le dépôt au chemin Python
    sys.path.insert(0, os.path.abspath("Wan2_1"))
    
    try:
        # Utiliser un chemin avec plus d'espace disponible
        temp_model_dir = "/workspace/temp_model"
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)
        os.makedirs(temp_model_dir, exist_ok=True)
        
        # Créer des liens symboliques pour les fichiers volumineux du modèle de base
        logger.info(f"Création de liens symboliques pour les fichiers du modèle de base depuis {base_model_dir}...")
        for item in os.listdir(base_model_dir):
            s = os.path.join(base_model_dir, item)
            d = os.path.join(temp_model_dir, item)
            
            # Utiliser des liens symboliques pour les fichiers volumineux
            if item.endswith('.safetensors') or item.endswith('.pth'):
                logger.info(f"Création d'un lien symbolique pour {item}")
                os.symlink(os.path.abspath(s), d)
            elif os.path.isdir(s):
                shutil.copytree(s, d, symlinks=True, dirs_exist_ok=True)
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
                
                # Ne pas écraser les liens symboliques existants pour les fichiers volumineux
                if os.path.exists(d) and (item.endswith('.safetensors') or item.endswith('.pth')):
                    logger.info(f"Fichier {item} déjà présent, conservation du lien symbolique")
                    continue
                
                if os.path.isdir(s):
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
        
        # Créer un dossier lora dans le dossier temporaire
        lora_dest = os.path.join(temp_model_dir, "lora")
        os.makedirs(lora_dest, exist_ok=True)
        
        # Copier tous les fichiers LoRA trouvés
        for lora_file in lora_files:
            lora_filename = os.path.basename(lora_file)
            dest_path = os.path.join(lora_dest, lora_filename)
            logger.info(f"Copie du fichier LoRA {lora_file} vers {dest_path}")
            shutil.copy2(lora_file, dest_path)
        
        # Ajouter le trigger word au prompt si spécifié
        if trigger_word:
            logger.info(f"Ajout du trigger word '{trigger_word}' au prompt")
            prompt = f"{trigger_word}, {prompt}"
        
        # Créer un fichier de configuration pour l'inférence
        config_file = os.path.join(temp_model_dir, "inference_config.json")
        import json
        with open(config_file, "w") as f:
            json.dump({
                "prompt": prompt,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
                "lora_weight": lora_weight
            }, f)
        
        # Importer le module generate
        try:
            logger.info("Importation du module generate...")
            from Wan2_1.generate import generate
            
            # Examiner la signature de la fonction generate
            import inspect
            logger.info(f"Signature de la fonction generate: {inspect.signature(generate)}")
            
            # Générer l'image
            logger.info(f"Génération d'image avec le prompt: {prompt}")
            logger.info(f"Utilisation du modèle dans: {temp_model_dir}")
            
            # Essayer différentes façons d'appeler la fonction generate
            try:
                # Première tentative: positional argument
                logger.info("Tentative d'appel avec argument positionnel...")
                output = generate(temp_model_dir, prompt)
            except Exception as e1:
                logger.warning(f"Première tentative échouée: {str(e1)}")
                try:
                    # Deuxième tentative: avec text=prompt
                    logger.info("Tentative d'appel avec text=prompt...")
                    output = generate(temp_model_dir, text=prompt)
                except Exception as e2:
                    logger.warning(f"Deuxième tentative échouée: {str(e2)}")
                    try:
                        # Troisième tentative: avec juste le modèle
                        logger.info("Tentative d'appel avec juste le modèle...")
                        # Créer un fichier temporaire avec le prompt
                        prompt_file = os.path.join(temp_model_dir, "prompt.txt")
                        with open(prompt_file, "w") as f:
                            f.write(prompt)
                        output = generate(temp_model_dir)
                    except Exception as e3:
                        logger.error(f"Troisième tentative échouée: {str(e3)}")
                        
                        # Quatrième tentative: utiliser directement le script generate.py
                        logger.info("Tentative d'utilisation directe du script generate.py...")
                        try:
                            cmd = [
                                sys.executable,
                                os.path.join("Wan2_1", "generate.py"),
                                "--model_path", temp_model_dir,
                                "--prompt", prompt,
                                "--output_dir", output_dir
                            ]
                            logger.info(f"Exécution de la commande: {' '.join(cmd)}")
                            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                            logger.info(f"Résultat: {result.stdout}")
                            
                            # Chercher le chemin de l'image générée dans la sortie
                            import re
                            match = re.search(r"Image sauvegardée à: (.+)", result.stdout)
                            if match:
                                output_path = match.group(1)
                                logger.info(f"Image sauvegardée à: {output_path}")
                                return output_path
                            else:
                                logger.error("Impossible de trouver le chemin de l'image générée dans la sortie")
                                return None
                        except Exception as e4:
                            logger.error(f"Quatrième tentative échouée: {str(e4)}")
                            import traceback
                            logger.error(traceback.format_exc())
                            raise Exception("Toutes les tentatives d'appel à generate ont échoué")
            
            # Sauvegarder l'image générée
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:20])
            output_path = os.path.join(output_dir, f"{safe_prompt}_{timestamp}.png")
            
            if isinstance(output, list) and len(output) > 0 and hasattr(output[0], "save"):
                output[0].save(output_path)
                logger.info(f"Image sauvegardée à: {output_path}")
                return output_path
            elif hasattr(output, "save"):
                output.save(output_path)
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
            # Supprimer uniquement les fichiers non symboliques
            for item in os.listdir(temp_model_dir):
                path = os.path.join(temp_model_dir, item)
                if not os.path.islink(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
            # Supprimer le dossier lui-même
            shutil.rmtree(temp_model_dir)

def main():
    parser = argparse.ArgumentParser(description="Inférence avec le modèle fine-tuné")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt pour la génération d'image")
    parser.add_argument("--model_dir", type=str, default="./hf_model_export", help="Chemin vers le modèle fine-tuné")
    parser.add_argument("--base_model_dir", type=str, default="./Wan2.1-T2V-14B", help="Chemin vers le modèle de base")
    parser.add_argument("--output_dir", type=str, default="./generated_outputs", help="Dossier de sortie")
    parser.add_argument("--steps", type=int, default=30, help="Nombre d'étapes d'inférence")
    parser.add_argument("--guidance", type=float, default=7.5, help="Échelle de guidance")
    parser.add_argument("--lora_weight", type=float, default=0.8, help="Poids du LoRA (entre 0 et 1)")
    parser.add_argument("--trigger_word", type=str, help="Mot déclencheur pour le LoRA")
    
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
        args.guidance,
        args.lora_weight,
        args.trigger_word
    )
    
    if output_path:
        logger.info(f"Inférence terminée avec succès. Image sauvegardée à: {output_path}")
    else:
        logger.error("Échec de l'inférence")

if __name__ == "__main__":
    main() 
