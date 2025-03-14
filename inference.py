import torch
from diffusers import DiffusionPipeline
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
        logging.FileHandler("inference.log")
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

def load_module_from_file(file_path, module_name):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        return None
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def download_model_weights(model_name="Wan-AI/Wan2.1-T2V-14B", local_dir="./Wan2.1-T2V-14B"):
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

def initialize_model(model_path, max_seq_length=2048, load_in_4bit=True, device=None):
    """
    Initialise le modèle et le tokenizer à partir du chemin spécifié.
    
    Args:
        model_path: Chemin vers le modèle sauvegardé
        max_seq_length: Longueur maximale de séquence
        load_in_4bit: Utiliser la quantification 4-bit
        device: Périphérique à utiliser (None pour auto-détection)
        
    Returns:
        tuple: (model, tokenizer, inference_config)
    """
    start_time = time.time()
    logger.info(f"Chargement du modèle depuis {model_path}")
    
    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le chemin du modèle {model_path} n'existe pas")
    
    # Déterminer le type de données à utiliser
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Déterminer le dtype en fonction du matériel
    if device == "cuda":
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere+ (RTX 30xx, A100, etc.)
            dtype = torch.bfloat16
            logger.info("Utilisation de bfloat16 sur GPU Ampere+")
        else:  # Architectures plus anciennes
            dtype = torch.float16
            logger.info("Utilisation de float16 sur GPU pré-Ampere")
    else:
        dtype = torch.float32
        logger.info("Utilisation de float32 sur CPU")
        load_in_4bit = False  # Désactiver la quantification 4-bit sur CPU
    
    # Charger le fichier de configuration d'inférence s'il existe
    config_path = os.path.join(model_path, "inference_config.json")
    inference_config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                inference_config = json.load(f)
                logger.info(f"Configuration d'inférence chargée: {inference_config}")
        except Exception as e:
            logger.warning(f"Erreur lors du chargement de la configuration d'inférence: {e}")
    
    # Vérifier si nous avons un modèle de diffusion (Stable Diffusion)
    is_diffusion_model = False
    if os.path.exists(os.path.join(model_path, "unet")) or os.path.exists(os.path.join(model_path, "unet_lora")):
        is_diffusion_model = True
        logger.info("Détection d'un modèle de diffusion (Stable Diffusion)")
    
    # Vérifier si nous avons un modèle LoRA
    is_lora_model = False
    if os.path.exists(os.path.join(model_path, "unet_lora")) or os.path.exists(os.path.join(model_path, "text_encoder_lora")):
        is_lora_model = True
        logger.info("Détection d'adaptateurs LoRA")
    
    # Charger le modèle en fonction de son type
    if is_diffusion_model:
        # Pour les modèles de diffusion (Stable Diffusion)
        from diffusers import StableDiffusionPipeline, DiffusionPipeline
        
        if is_lora_model:
            # Charger le modèle de base puis appliquer les adaptateurs LoRA
            logger.info("Chargement du modèle de base Stable Diffusion...")
            
            try:
                # Charger le modèle de base (Stable Diffusion 1.5 par défaut)
                base_model_id = "runwayml/stable-diffusion-v1-5"
                logger.info(f"Utilisation du modèle de base: {base_model_id}")
                
                # Charger le pipeline de base
                pipe = StableDiffusionPipeline.from_pretrained(
                    base_model_id,
                    torch_dtype=dtype
                )
                pipe = pipe.to(device)
                
                # Charger et appliquer les adaptateurs LoRA
                logger.info("Application des adaptateurs LoRA...")
                
                # Charger les adaptateurs LoRA pour l'UNet
                if os.path.exists(os.path.join(model_path, "unet_lora")):
                    from diffusers import UNet2DConditionModel
                    
                    # Charger l'adaptateur LoRA pour l'UNet
                    unet_lora_path = os.path.join(model_path, "unet_lora")
                    logger.info(f"Chargement de l'adaptateur LoRA pour l'UNet depuis {unet_lora_path}")
                    
                    # Appliquer l'adaptateur LoRA à l'UNet
                    pipe.unet.load_attn_procs(unet_lora_path)
                    logger.info("Adaptateur LoRA appliqué à l'UNet")
                
                # Charger les adaptateurs LoRA pour l'encodeur de texte
                if os.path.exists(os.path.join(model_path, "text_encoder_lora")):
                    from diffusers import CLIPTextModel
                    
                    # Charger l'adaptateur LoRA pour l'encodeur de texte
                    text_encoder_lora_path = os.path.join(model_path, "text_encoder_lora")
                    logger.info(f"Chargement de l'adaptateur LoRA pour l'encodeur de texte depuis {text_encoder_lora_path}")
                    
                    # Appliquer l'adaptateur LoRA à l'encodeur de texte
                    pipe.text_encoder.load_attn_procs(text_encoder_lora_path)
                    logger.info("Adaptateur LoRA appliqué à l'encodeur de texte")
                
                # Utiliser le tokenizer du pipeline
                tokenizer = pipe.tokenizer
                logger.info("Tokenizer chargé depuis le pipeline")
                
                model = pipe
                logger.info("Pipeline avec adaptateurs LoRA chargé avec succès")
                
            except Exception as e:
                logger.error(f"Erreur lors du chargement des adaptateurs LoRA: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
        else:
            # Charger un modèle de diffusion complet
            logger.info("Chargement d'un modèle de diffusion complet...")
            
            try:
                # Charger le pipeline complet
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype
                )
                pipe = pipe.to(device)
                
                # Utiliser le tokenizer du pipeline
                tokenizer = pipe.tokenizer
                
                model = pipe
                logger.info("Pipeline de diffusion chargé avec succès")
                
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle de diffusion: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
    else:
        # Pour les modèles de langage (LLM)
        try:
            from unsloth import FastLanguageModel
            logger.info("Utilisation de Unsloth FastLanguageModel pour le chargement")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
                device_map="auto" if device == "cuda" else device
            )
        except ImportError:
            # Fallback à la méthode standard si Unsloth n'est pas disponible
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("Unsloth non disponible, utilisation de AutoModelForCausalLM standard")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                load_in_4bit=load_in_4bit,
                device_map="auto" if device == "cuda" else device
            )
    
    # Mettre le modèle en mode évaluation
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "unet") and hasattr(model.unet, "eval"):
        model.unet.eval()
    
    logger.info(f"Modèle chargé avec succès en {time.time() - start_time:.2f} secondes")
    return model, tokenizer, inference_config

def generate_response(model, tokenizer, prompt, inference_config=None, **kwargs):
    """
    Génère une réponse à partir d'un prompt.
    
    Args:
        model: Le modèle à utiliser
        tokenizer: Le tokenizer à utiliser
        prompt: Le prompt à utiliser
        inference_config: Configuration d'inférence (optionnel)
        **kwargs: Paramètres supplémentaires pour la génération
        
    Returns:
        str: La réponse générée
    """
    # Paramètres par défaut
    generation_config = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "num_beams": 1,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 2,
        "use_cache": True
    }
    
    # Mettre à jour avec la configuration d'inférence si disponible
    if inference_config:
        for key in ["max_new_tokens", "temperature", "top_p"]:
            if key in inference_config:
                generation_config[key] = inference_config[key]
    
    # Mettre à jour avec les paramètres fournis
    generation_config.update(kwargs)
    
    # Encoder le prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device if hasattr(model, "device") else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Générer la réponse
    with torch.no_grad():
        # Vérifier si nous avons un modèle Wan ou un modèle standard
        if hasattr(model, "unet") and not hasattr(model, "generate"):
            # Pour les modèles Wan, nous devons implémenter une logique spécifique
            logger.info("Génération avec un modèle Wan (non implémentée)")
            # Cette partie devrait être adaptée en fonction des besoins spécifiques
            output_text = "La génération avec ce modèle Wan n'est pas encore implémentée."
        else:
            # Pour les modèles standard
            output_ids = model.generate(
                input_ids=input_ids,
                **generation_config
            )
            
            # Décoder la réponse (en ignorant le prompt)
            output_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    return output_text

def interactive_evaluation(model, tokenizer, inference_config=None):
    """
    Mode interactif pour évaluer le modèle.
    
    Args:
        model: Le modèle à utiliser
        tokenizer: Le tokenizer à utiliser
        inference_config: Configuration d'inférence (optionnel)
    """
    prompt_template = """Tu es un expert en fiscalité, comptabilité, ton objectif et d'accompagner les entreprises dans leur enjeux. 

### Texte principal:
{texte}

### Question:
{question}

### Réponse:
"""
    
    print("\n" + "="*80)
    print("Mode interactif d'évaluation du modèle comptable")
    print("Tapez 'exit' à tout moment pour quitter")
    print("="*80 + "\n")
    
    while True:
        try:
            texte = input("\nEntrez le texte principal (ou 'exit' pour quitter) : ")
            if texte.lower() == 'exit':
                break
                
            question = input("Entrez la question : ")
            if question.lower() == 'exit':
                break
            
            # Construire le prompt
            prompt = prompt_template.format(
                texte=texte,
                question=question
            )
            
            print("\nGénération de la réponse...")
            start_time = time.time()
            
            # Générer la réponse
            output_text = generate_response(model, tokenizer, prompt, inference_config)
            
            # Afficher la réponse
            print("\n" + "-"*80)
            print("RÉPONSE:")
            print(output_text)
            print("-"*80)
            print(f"Temps de génération: {time.time() - start_time:.2f} secondes")
            
        except KeyboardInterrupt:
            print("\nInterruption détectée. Sortie du mode interactif.")
            break
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            print(f"\nUne erreur s'est produite: {e}")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Inférence avec un modèle fine-tuné")
    parser.add_argument("--model_path", type=str, default="/workspace/full_model", 
                        help="Chemin vers le modèle sauvegardé (par défaut: modèle complet)")
    parser.add_argument("--use_lora_only", action="store_true",
                        help="Utiliser uniquement les poids LoRA (hf_model_export) au lieu du modèle complet")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Longueur maximale de séquence")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Utiliser la quantification 4-bit")
    parser.add_argument("--cpu", action="store_true",
                        help="Forcer l'utilisation du CPU")
    parser.add_argument("--prompt", type=str,
                        help="Prompt à utiliser pour la génération (si non fourni, mode interactif)")
    parser.add_argument("--output_dir", type=str, default="./generated_outputs",
                        help="Dossier de sortie pour les images générées")
    parser.add_argument("--size", type=str, default="832*480",
                        help="Taille de l'image générée (format: largeur*hauteur)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Nombre d'étapes d'inférence pour les modèles de diffusion")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Échelle de guidance pour les modèles de diffusion")
    
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
    if not os.path.exists(model_path):
        logger.error(f"Le chemin du modèle {model_path} n'existe pas")
        print(f"ERREUR: Le chemin du modèle '{model_path}' n'existe pas.")
        print("Veuillez spécifier un chemin valide avec --model_path ou utiliser --use_lora_only")
        return 1
    
    # Déterminer le périphérique
    device = "cpu" if args.cpu else None
    
    try:
        # Initialiser le modèle
        model, tokenizer, inference_config = initialize_model(
            model_path, 
            args.max_seq_length,
            args.load_in_4bit,
            device
        )
        
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Mode interactif ou génération unique
        if args.prompt:
            logger.info(f"Génération avec le prompt: {args.prompt}")
            
            # Traiter la taille si spécifiée
            size = None
            if args.size:
                try:
                    width, height = map(int, args.size.split('*'))
                    size = (width, height)
                    logger.info(f"Taille d'image spécifiée: {width}x{height}")
                except:
                    logger.warning(f"Format de taille invalide: {args.size}, utilisation de la taille par défaut")
            
            # Générer la réponse
            try:
                # Vérifier si nous avons un modèle de diffusion (StableDiffusionPipeline)
                if hasattr(model, 'unet') and hasattr(model, 'vae') and hasattr(model, 'text_encoder'):
                    # Pour les modèles de diffusion (Stable Diffusion)
                    logger.info("Utilisation du mode de génération d'image avec Stable Diffusion")
                    print("Génération d'image en cours...")
                    
                    # Préparer les paramètres d'inférence
                    inference_params = {
                        "prompt": args.prompt,
                        "num_inference_steps": args.num_inference_steps,
                        "guidance_scale": args.guidance_scale,
                    }
                    
                    # Ajouter la taille si spécifiée
                    if size:
                        inference_params["width"] = size[0]
                        inference_params["height"] = size[1]
                    
                    # Générer l'image
                    output = model(**inference_params)
                    
                    # Sauvegarder l'image
                    timestamp = int(time.time())
                    output_file = os.path.join(args.output_dir, f"image_{timestamp}.png")
                    output.images[0].save(output_file)
                    
                    logger.info(f"Image générée et sauvegardée dans {output_file}")
                    print(f"Image générée et sauvegardée dans {output_file}")
                    
                elif hasattr(model, 'generate') and callable(model.generate):
                    # Pour les modèles texte standard
                    response = generate_response(model, tokenizer, args.prompt, inference_config)
                    
                    # Sauvegarder la réponse dans un fichier texte
                    output_file = os.path.join(args.output_dir, f"response_{int(time.time())}.txt")
                    with open(output_file, 'w') as f:
                        f.write(response)
                    logger.info(f"Réponse sauvegardée dans {output_file}")
                    print(f"Réponse générée et sauvegardée dans {output_file}")
                else:
                    logger.error("Type de modèle non reconnu pour la génération")
                    print("ERREUR: Type de modèle non reconnu pour la génération")
            except Exception as e:
                logger.error(f"Erreur lors de la génération: {e}")
                print(f"ERREUR lors de la génération: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            # Mode interactif
            if hasattr(model, 'unet') and hasattr(model, 'vae') and hasattr(model, 'text_encoder'):
                print("\n" + "="*80)
                print("Mode interactif d'évaluation du modèle de diffusion")
                print("Tapez 'exit' à tout moment pour quitter")
                print("="*80 + "\n")
                
                while True:
                    try:
                        prompt = input("\nEntrez un prompt (ou 'exit' pour quitter) : ")
                        if prompt.lower() == 'exit':
                            break
                        
                        print("\nGénération de l'image...")
                        start_time = time.time()
                        
                        # Générer l'image
                        output = model(
                            prompt=prompt,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale
                        )
                        
                        # Sauvegarder l'image
                        timestamp = int(time.time())
                        output_file = os.path.join(args.output_dir, f"image_{timestamp}.png")
                        output.images[0].save(output_file)
                        
                        print(f"\nImage générée et sauvegardée dans {output_file}")
                        print(f"Temps de génération: {time.time() - start_time:.2f} secondes")
                        
                    except KeyboardInterrupt:
                        print("\nInterruption détectée. Sortie du mode interactif.")
                        break
                    except Exception as e:
                        logger.error(f"Erreur lors de la génération: {e}")
                        print(f"\nUne erreur s'est produite: {e}")
            else:
                # Mode interactif pour les modèles de texte
                interactive_evaluation(model, tokenizer, inference_config)
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        print(f"ERREUR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
