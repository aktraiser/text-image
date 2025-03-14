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
    
    # Vérifier si nous avons un modèle Wan2.1
    is_wan_model = False
    if os.path.exists(os.path.join(model_path, "unet")) or "Wan" in model_path:
        is_wan_model = True
        logger.info("Détection d'un modèle Wan2.1")
    
    # Charger le modèle en fonction de son type
    if is_wan_model:
        # Cloner le dépôt Wan2.1 si nécessaire
        clone_wan_repository()
        
        try:
            # Essayer de charger le module generate.py directement
            generate_module = load_module_from_file(os.path.join("Wan2_1", "generate.py"), "generate")
            
            if generate_module and hasattr(generate_module, "load_t2v_pipeline"):
                # Charger le modèle avec la fonction personnalisée
                model = generate_module.load_t2v_pipeline(
                    task="t2v-14B",
                    ckpt_dir=model_path,
                    size="1280*720",  # Résolution par défaut
                    device=device,
                    dtype=dtype
                )
                logger.info("Modèle chargé avec succès via la pipeline personnalisée")
                
                # Charger le tokenizer
                from transformers import AutoTokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
                    logger.info("Tokenizer chargé avec succès")
                except Exception:
                    logger.warning("Impossible de charger le tokenizer, utilisation de t5-base comme fallback")
                    tokenizer = AutoTokenizer.from_pretrained("t5-base")
            else:
                raise ImportError("Impossible de trouver la fonction load_t2v_pipeline")
                
        except (ImportError, FileNotFoundError) as e:
            # Fallback à une approche directe
            logger.info(f"Utilisation de la méthode de fallback pour charger le modèle: {str(e)}")
            
            # Créer un wrapper simple autour du modèle
            from transformers import AutoTokenizer
            from diffusers import UNet2DConditionModel
            
            class WanModelWrapper:
                def __init__(self, model_dir):
                    self.model_dir = model_dir
                    self.unet = None
                    self.tokenizer = None
                    
                    # Essayer de charger le composant UNet directement
                    try:
                        self.unet = UNet2DConditionModel.from_pretrained(
                            os.path.join(model_dir, "unet"),
                            torch_dtype=dtype
                        )
                        logger.info("Composant UNet chargé avec succès")
                    except Exception as unet_error:
                        logger.warning(f"Impossible de charger l'UNet: {str(unet_error)}")
                    
                    # Charger le tokenizer
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
                        logger.info("Tokenizer chargé avec succès")
                    except Exception:
                        logger.warning("Impossible de charger le tokenizer, utilisation de t5-base comme fallback")
                        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
                
                def generate(self, **kwargs):
                    # Implémentation simplifiée de la génération
                    logger.info("Génération avec le modèle Wan")
                    # Cette méthode devrait être adaptée en fonction des besoins spécifiques
                    return kwargs.get("input_ids", None)
            
            model = WanModelWrapper(model_path)
            tokenizer = model.tokenizer
    else:
        # Charger un modèle standard avec Unsloth si disponible
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
    parser = argparse.ArgumentParser(description="Inférence avec un modèle Wan2.1 fine-tuné")
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
    
    args = parser.parse_args()
    
    # Déterminer le chemin du modèle
    model_path = args.model_path
    if args.use_lora_only:
        model_path = "hf_model_export"
        logger.info(f"Utilisation des poids LoRA uniquement depuis {model_path}")
    
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
                # Adapter en fonction du type de modèle
                if hasattr(model, 'generate') and callable(model.generate):
                    # Pour les modèles texte standard
                    response = generate_response(model, tokenizer, args.prompt, inference_config)
                    
                    # Sauvegarder la réponse dans un fichier texte
                    output_file = os.path.join(args.output_dir, f"response_{int(time.time())}.txt")
                    with open(output_file, 'w') as f:
                        f.write(response)
                    logger.info(f"Réponse sauvegardée dans {output_file}")
                    
                elif hasattr(model, 'unet') or 'Wan' in model_path:
                    # Pour les modèles de génération d'images/vidéos
                    logger.info("Utilisation du mode de génération d'image/vidéo")
                    
                    # Essayer d'utiliser la fonction de génération spécifique à Wan2.1 si disponible
                    try:
                        sys.path.append(os.path.abspath("Wan2_1"))
                        from Wan2_1.generate import generate_video
                        
                        output_file = os.path.join(args.output_dir, f"output_{int(time.time())}.mp4")
                        generate_video(
                            model=model,
                            prompt=args.prompt,
                            output_path=output_file,
                            size=size
                        )
                        logger.info(f"Vidéo générée et sauvegardée dans {output_file}")
                    except (ImportError, AttributeError) as e:
                        logger.error(f"Erreur lors de la génération avec Wan2.1: {e}")
                        logger.info("Tentative de génération avec pipeline standard...")
                        
                        # Fallback à une pipeline standard de diffusion
                        pipeline = DiffusionPipeline.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                        )
                        pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
                        
                        output_file = os.path.join(args.output_dir, f"image_{int(time.time())}.png")
                        image = pipeline(args.prompt).images[0]
                        image.save(output_file)
                        logger.info(f"Image générée et sauvegardée dans {output_file}")
                else:
                    logger.error("Type de modèle non reconnu pour la génération")
            except Exception as e:
                logger.error(f"Erreur lors de la génération: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            # Mode interactif
            interactive_evaluation(model, tokenizer, inference_config)
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
