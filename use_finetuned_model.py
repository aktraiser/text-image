import os
import sys
import logging
import argparse
import datetime
import subprocess
import shutil
import glob

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_lora_files():
    """Recherche les fichiers LoRA dans les dossiers courants."""
    lora_files = []
    
    # Rechercher dans le dossier outputs/lora
    if os.path.exists("outputs/lora"):
        lora_files.extend(glob.glob("outputs/lora/**/*.safetensors", recursive=True))
    
    # Rechercher dans le dossier outputs/checkpoint-*
    lora_files.extend(glob.glob("outputs/checkpoint-*/*.safetensors", recursive=True))
    
    # Rechercher dans le dossier hf_model_export
    if os.path.exists("hf_model_export"):
        lora_files.extend(glob.glob("hf_model_export/**/*.safetensors", recursive=True))
    
    return lora_files

def run_inference(prompt, model_dir="./hf_model_export", base_model_dir="./Wan2.1-T2V-14B", 
                 output_dir="./generated_outputs", num_steps=30, guidance_scale=7.5,
                 lora_weight=0.8, trigger_word=None, size="832*480"):
    """Exécute l'inférence en utilisant directement le script generate.py."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Cloner le dépôt si nécessaire
    if not os.path.exists("Wan2_1"):
        logger.info("Clonage du dépôt Wan2.1...")
        subprocess.run(["git", "clone", "https://github.com/Wan-Video/Wan2.1.git", "Wan2_1"], check=True)
    
    # Rechercher les fichiers LoRA
    lora_files = find_lora_files()
    if lora_files:
        logger.info(f"Fichiers LoRA trouvés: {lora_files}")
        # Utiliser uniquement le dernier checkpoint
        lora_files = sorted(lora_files)
        lora_file = lora_files[-1]
        logger.info(f"Utilisation du fichier LoRA: {lora_file}")
    else:
        logger.warning("Aucun fichier LoRA trouvé")
        lora_file = None
    
    # Ajouter le trigger word au prompt si spécifié
    if trigger_word:
        logger.info(f"Ajout du trigger word '{trigger_word}' au prompt")
        prompt = f"{trigger_word}, {prompt}"
    
    # Préparer la commande
    cmd = [
        sys.executable,
        "Wan2_1/generate.py",
        "--task", "t2v-14B",
        "--size", size,
        "--ckpt_dir", base_model_dir,
        "--prompt", prompt,
        "--output_dir", output_dir,
        "--num_inference_steps", str(num_steps),
        "--guidance_scale", str(guidance_scale)
    ]
    
    # Ajouter le fichier LoRA s'il existe
    if lora_file:
        cmd.extend(["--lora_path", lora_file, "--lora_scale", str(lora_weight)])
    
    # Exécuter la commande
    logger.info(f"Exécution de la commande: {' '.join(cmd)}")
    try:
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
            # Si on ne trouve pas le chemin dans la sortie, chercher le fichier le plus récent
            files = glob.glob(f"{output_dir}/*.mp4") + glob.glob(f"{output_dir}/*.gif") + glob.glob(f"{output_dir}/*.png")
            if files:
                latest_file = max(files, key=os.path.getctime)
                logger.info(f"Fichier le plus récent trouvé: {latest_file}")
                return latest_file
            else:
                logger.error("Aucun fichier généré trouvé")
                return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution de la commande: {str(e)}")
        logger.error(f"Sortie standard: {e.stdout}")
        logger.error(f"Sortie d'erreur: {e.stderr}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Inférence directe avec le modèle Wan2.1")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt pour la génération de vidéo")
    parser.add_argument("--model_dir", type=str, default="./hf_model_export", help="Chemin vers le modèle fine-tuné")
    parser.add_argument("--base_model_dir", type=str, default="./Wan2.1-T2V-14B", help="Chemin vers le modèle de base")
    parser.add_argument("--output_dir", type=str, default="./generated_outputs", help="Dossier de sortie")
    parser.add_argument("--steps", type=int, default=30, help="Nombre d'étapes d'inférence")
    parser.add_argument("--guidance", type=float, default=7.5, help="Échelle de guidance")
    parser.add_argument("--lora_weight", type=float, default=0.8, help="Poids du LoRA (entre 0 et 1)")
    parser.add_argument("--trigger_word", type=str, help="Mot déclencheur pour le LoRA")
    parser.add_argument("--size", type=str, default="832*480", help="Taille de la vidéo (832*480 ou 1280*720)")
    
    args = parser.parse_args()
    
    # Exécuter l'inférence
    output_path = run_inference(
        args.prompt,
        args.model_dir,
        args.base_model_dir,
        args.output_dir,
        args.steps,
        args.guidance,
        args.lora_weight,
        args.trigger_word,
        args.size
    )
    
    if output_path:
        logger.info(f"Inférence terminée avec succès. Fichier sauvegardé à: {output_path}")
    else:
        logger.error("Échec de l'inférence")

if __name__ == "__main__":
    main() 
