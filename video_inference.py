import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, TextToVideoSDPipeline
import json
import os
import argparse
import logging
import time
from pathlib import Path
import sys
import importlib.util
import subprocess
from huggingface_hub import snapshot_download

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("video_inference.log")
    ]
)
logger = logging.getLogger(__name__)

# Afficher un message d'information sur les chemins de modèle
logger.info("""
==========================================================
INFORMATION SUR LES CHEMINS DE MODÈLE:
- Modèle complet: /workspace/full_model (recommandé)
- Poids LoRA uniquement: hf_model_export
==========================================================
""")

# Désactiver le parallélisme des tokenizers pour éviter les warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Libérer la mémoire GPU avant de commencer
torch.cuda.empty_cache()

def clone_wan_repository():
    """Clone the Wan2.1 repository to access the model code."""
    logger.info("Cloning the Wan2.1 repository...")
    
    if not os.path.exists("Wan2_1"):  # Changed directory name to avoid the dot
        subprocess.run(["git", "clone", "https://github.com/Wan-Video/Wan2.1.git", "Wan2_1"], check=True)
        logger.info("Repository cloned successfully.")
    else:
        logger.info("Repository already exists, skipping clone.")
    
    # Add the repository to Python path to import modules
    sys.path.append(os.path.abspath("Wan2_1"))

def download_model_weights(model_name="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", local_dir="./Wan2.1-I2V-14B"):
    """Download the model weights from Hugging Face."""
    logger.info(f"Downloading model weights for {model_name}...")
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        logger.info(f"Model weights downloaded to {local_dir}")
    else:
        logger.info(f"Model weights directory {local_dir} already exists, skipping download.")
    
    return local_dir

def initialize_video_model(model_path, use_fp16=True, device=None):
    """
    Initialise le modèle de génération de vidéo à partir du chemin spécifié.
    
    Args:
        model_path: Chemin vers le modèle ou ID du modèle Hugging Face
        use_fp16: Utiliser la précision FP16
        device: Périphérique à utiliser (None pour auto-détection)
        
    Returns:
        pipeline: Pipeline de génération de vidéo
    """
    start_time = time.time()
    logger.info(f"Chargement du modèle de génération de vidéo depuis {model_path}")
    
    # Déterminer le type de données à utiliser
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Déterminer le dtype en fonction du matériel et des paramètres
    if device == "cuda" and use_fp16:
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere+ (RTX 30xx, A100, etc.)
            dtype = torch.bfloat16
            logger.info("Utilisation de bfloat16 sur GPU Ampere+")
        else:  # Architectures plus anciennes
            dtype = torch.float16
            logger.info("Utilisation de float16 sur GPU pré-Ampere")
    else:
        dtype = torch.float32
        logger.info("Utilisation de float32")
    
    try:
        # Vérifier si le modèle est un chemin local ou un ID Hugging Face
        if os.path.exists(model_path):
            logger.info(f"Chargement du modèle depuis le chemin local: {model_path}")
            # Déterminer le type de modèle en fonction des fichiers présents
            if os.path.exists(os.path.join(model_path, "unet_lora")) or os.path.exists(os.path.join(model_path, "text_encoder_lora")):
                # C'est un modèle LoRA, nous devons l'appliquer à un modèle de base
                logger.info("Détection d'adaptateurs LoRA, chargement du modèle de base...")
                
                # Charger le modèle de base pour la génération de vidéo
                base_model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
                logger.info(f"Utilisation du modèle de base: {base_model_id}")
                
                # Charger le pipeline de base
                pipeline = TextToVideoSDPipeline.from_pretrained(
                    base_model_id,
                    torch_dtype=dtype,
                    variant="fp16" if use_fp16 else None
                )
                
                # Appliquer les adaptateurs LoRA
                logger.info("Application des adaptateurs LoRA...")
                
                # Charger et appliquer les adaptateurs LoRA pour l'UNet
                if os.path.exists(os.path.join(model_path, "unet_lora")):
                    from peft import PeftModel
                    
                    # Charger l'adaptateur LoRA pour l'UNet
                    unet_lora_path = os.path.join(model_path, "unet_lora")
                    logger.info(f"Chargement de l'adaptateur LoRA pour l'UNet depuis {unet_lora_path}")
                    
                    # Appliquer l'adaptateur LoRA à l'UNet
                    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, unet_lora_path)
                    logger.info("Adaptateur LoRA appliqué à l'UNet")
                
                # Charger les adaptateurs LoRA pour l'encodeur de texte
                if os.path.exists(os.path.join(model_path, "text_encoder_lora")):
                    from peft import PeftModel
                    
                    # Charger l'adaptateur LoRA pour l'encodeur de texte
                    text_encoder_lora_path = os.path.join(model_path, "text_encoder_lora")
                    logger.info(f"Chargement de l'adaptateur LoRA pour l'encodeur de texte depuis {text_encoder_lora_path}")
                    
                    # Appliquer l'adaptateur LoRA à l'encodeur de texte
                    pipeline.text_encoder = PeftModel.from_pretrained(pipeline.text_encoder, text_encoder_lora_path)
                    logger.info("Adaptateur LoRA appliqué à l'encodeur de texte")
            else:
                # C'est un modèle complet
                logger.info("Chargement d'un modèle complet...")
                pipeline = TextToVideoSDPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    variant="fp16" if use_fp16 else None
                )
        else:
            # C'est un ID de modèle Hugging Face
            logger.info(f"Chargement du modèle depuis Hugging Face: {model_path}")
            pipeline = TextToVideoSDPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                variant="fp16" if use_fp16 else None
            )
        
        # Activer l'offload du modèle sur CPU pour économiser de la mémoire
        if hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
        else:
            # Fallback pour les versions plus anciennes
            pipeline = pipeline.to(device)
        
        # Activer les techniques d'optimisation de mémoire
        if hasattr(pipeline, "vae"):
            pipeline.vae.enable_slicing()
            if hasattr(pipeline.vae, "enable_tiling"):
                pipeline.vae.enable_tiling()
        
        # Activer le slicing d'attention si disponible
        if hasattr(pipeline, "unet"):
            if hasattr(pipeline.unet, "enable_attention_slicing"):
                pipeline.unet.enable_attention_slicing()
        
        logger.info(f"Modèle chargé avec succès en {time.time() - start_time:.2f} secondes")
        return pipeline
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def generate_video(pipeline, prompt, negative_prompt=None, num_frames=24, num_inference_steps=50, 
                  guidance_scale=7.5, height=480, width=832, fps=8, seed=None, output_dir="./generated_videos"):
    """
    Génère une vidéo à partir d'un prompt textuel.
    
    Args:
        pipeline: Pipeline de génération de vidéo
        prompt: Prompt textuel pour la génération
        negative_prompt: Prompt négatif pour éviter certains éléments
        num_frames: Nombre d'images à générer
        num_inference_steps: Nombre d'étapes d'inférence
        guidance_scale: Échelle de guidance
        height: Hauteur de la vidéo
        width: Largeur de la vidéo
        fps: Images par seconde pour la vidéo
        seed: Graine aléatoire pour la reproductibilité
        output_dir: Dossier de sortie pour la vidéo générée
        
    Returns:
        str: Chemin vers la vidéo générée
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Définir la graine aléatoire si fournie
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = torch.Generator(device="cuda").manual_seed(int(time.time()))
    
    logger.info(f"Génération de vidéo avec le prompt: '{prompt}'")
    start_time = time.time()
    
    try:
        # Générer la vidéo
        from diffusers.utils import export_to_video
        
        # Adapter les paramètres en fonction du type de pipeline
        if hasattr(pipeline, "num_frames"):
            # Pour les pipelines qui supportent directement num_frames
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_frames=num_frames,
                height=height,
                width=width,
                generator=generator,
            )
        else:
            # Pour les pipelines qui utilisent un autre paramètre ou format
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )
        
        # Récupérer les frames du résultat
        if hasattr(result, "frames") and isinstance(result.frames, list):
            frames = result.frames[0]
        elif hasattr(result, "videos") and isinstance(result.videos, list):
            frames = result.videos[0]
        else:
            # Fallback pour d'autres formats de résultat
            frames = result.images
        
        # Générer un nom de fichier unique basé sur le timestamp
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")
        
        # Exporter les frames en vidéo
        export_to_video(frames, output_path, fps=fps)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Génération de vidéo terminée en {elapsed_time:.2f} secondes")
        logger.info(f"Vidéo sauvegardée à: {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la vidéo: {str(e)}")
        raise

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Génération de vidéo à partir d'un prompt")
    parser.add_argument("--model_path", type=str, default="/workspace/full_model", 
                        help="Chemin vers le modèle sauvegardé ou ID du modèle Hugging Face")
    parser.add_argument("--use_lora_only", action="store_true",
                        help="Utiliser uniquement les poids LoRA (hf_model_export) au lieu du modèle complet")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt textuel pour la génération de vidéo")
    parser.add_argument("--negative_prompt", type=str, default="Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured",
                        help="Prompt négatif pour éviter certains éléments")
    parser.add_argument("--output_dir", type=str, default="./generated_videos",
                        help="Dossier de sortie pour les vidéos générées")
    parser.add_argument("--num_frames", type=int, default=24,
                        help="Nombre d'images à générer")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Nombre d'étapes d'inférence")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Échelle de guidance")
    parser.add_argument("--size", type=str, default="832*480",
                        help="Taille de la vidéo générée (format: largeur*hauteur)")
    parser.add_argument("--fps", type=int, default=8,
                        help="Images par seconde pour la vidéo")
    parser.add_argument("--seed", type=int, default=None,
                        help="Graine aléatoire pour la reproductibilité")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Désactiver la précision FP16")
    
    args = parser.parse_args()
    
    # Afficher les arguments pour le débogage
    logger.info(f"Arguments reçus: {vars(args)}")
    
    # Déterminer le chemin du modèle
    model_path = args.model_path
    if args.use_lora_only:
        model_path = "hf_model_export"
        logger.info(f"Utilisation des poids LoRA uniquement depuis {model_path}")
    
    # Vérifier que le chemin du modèle existe
    logger.info(f"Vérification du chemin du modèle: {model_path}")
    if not os.path.exists(model_path) and not model_path.startswith(("Wan-AI/", "stabilityai/")):
        logger.error(f"Le chemin du modèle {model_path} n'existe pas")
        print(f"ERREUR: Le chemin du modèle '{model_path}' n'existe pas.")
        print("Veuillez spécifier un chemin valide avec --model_path ou utiliser --use_lora_only")
        return 1
    
    # Traiter la taille si spécifiée
    width, height = 832, 480
    if args.size:
        try:
            width, height = map(int, args.size.split('*'))
            logger.info(f"Taille de vidéo spécifiée: {width}x{height}")
        except:
            logger.warning(f"Format de taille invalide: {args.size}, utilisation de la taille par défaut")
    
    try:
        # Initialiser le modèle
        pipeline = initialize_video_model(
            model_path, 
            use_fp16=not args.no_fp16
        )
        
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Générer la vidéo
        output_path = generate_video(
            pipeline=pipeline,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=height,
            width=width,
            fps=args.fps,
            seed=args.seed,
            output_dir=args.output_dir
        )
        
        logger.info(f"Vidéo générée avec succès: {output_path}")
        print(f"Vidéo générée avec succès: {output_path}")
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        print(f"ERREUR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
