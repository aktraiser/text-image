import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import argparse
import logging
import time
from pathlib import Path
import pandas as pd

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

# Désactiver le parallélisme des tokenizers pour éviter les warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def initialize_model(model_path, max_seq_length=2048, load_in_4bit=True, device=None):
    """
    Initialise le modèle et le tokenizer à partir du chemin spécifié.
    
    Args:
        model_path: Chemin vers le modèle sauvegardé
        max_seq_length: Longueur maximale de séquence
        load_in_4bit: Utiliser la quantification 4-bit
        device: Périphérique à utiliser (None pour auto-détection)
        
    Returns:
        tuple: (model, tokenizer)
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
    
    # Charger le tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Tokenizer chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du tokenizer: {e}")
        raise
    
    # Charger le modèle
    try:
        # Vérifier si nous utilisons Unsloth
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
            logger.info("Unsloth non disponible, utilisation de AutoModelForCausalLM standard")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                load_in_4bit=load_in_4bit,
                device_map="auto" if device == "cuda" else device
            )
        
        logger.info(f"Modèle chargé avec succès en {time.time() - start_time:.2f} secondes")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise
    
    # Mettre le modèle en mode évaluation
    model.eval()
    
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
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    # Générer la réponse
    with torch.no_grad():
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
    prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Tu es un expert comptable spécialisé dans le conseil aux entreprises. Tu dois fournir une réponse professionnelle et précise basée uniquement sur le contexte fourni.

### Input:
Type: {content_type}
Sujet: {title}
Document: {main_text}
Question: {questions}
Source: {source}

### Response:
"""
    
    print("\n" + "="*80)
    print("Mode interactif d'évaluation du modèle comptable")
    print("Tapez 'exit' à tout moment pour quitter")
    print("="*80 + "\n")
    
    while True:
        try:
            content_type = input("\nType de contenu (ex: Rapport financier): ")
            if content_type.lower() == 'exit':
                break
                
            title = input("Sujet: ")
            if title.lower() == 'exit':
                break
                
            main_text = input("Document principal: ")
            if main_text.lower() == 'exit':
                break
                
            questions = input("Question: ")
            if questions.lower() == 'exit':
                break
                
            source = input("Source (optionnel): ")
            if source.lower() == 'exit':
                break
            
            # Construire le prompt
            prompt = prompt_template.format(
                content_type=content_type,
                title=title,
                main_text=main_text,
                questions=questions,
                source=source
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

def batch_evaluation_from_file(model, tokenizer, input_file, output_file=None, inference_config=None):
    """
    Évalue le modèle sur un lot de données à partir d'un fichier CSV.
    
    Args:
        model: Le modèle à utiliser
        tokenizer: Le tokenizer à utiliser
        input_file: Chemin vers le fichier CSV d'entrée
        output_file: Chemin vers le fichier CSV de sortie (optionnel)
        inference_config: Configuration d'inférence (optionnel)
    """
    # Vérifier si le fichier existe
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Le fichier d'entrée {input_file} n'existe pas")
    
    # Charger les données
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Fichier chargé avec succès: {len(df)} lignes")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier: {e}")
        raise
    
    # Vérifier les colonnes requises
    required_columns = ['content_type', 'title', 'main_text', 'questions']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le fichier CSV: {missing_columns}")
    
    # Ajouter une colonne 'source' si elle n'existe pas
    if 'source' not in df.columns:
        df['source'] = ""
    
    # Ajouter une colonne pour les réponses
    df['model_response'] = ""
    df['generation_time'] = 0.0
    
    # Template de prompt
    prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Tu es un expert comptable spécialisé dans le conseil aux entreprises. Tu dois fournir une réponse professionnelle et précise basée uniquement sur le contexte fourni.

### Input:
Type: {content_type}
Sujet: {title}
Document: {main_text}
Question: {questions}
Source: {source}

### Response:
"""
    
    # Générer les réponses
    total_start_time = time.time()
    for i, row in df.iterrows():
        try:
            # Construire le prompt
            prompt = prompt_template.format(
                content_type=row['content_type'],
                title=row['title'],
                main_text=row['main_text'],
                questions=row['questions'],
                source=row['source']
            )
            
            # Générer la réponse
            start_time = time.time()
            output_text = generate_response(model, tokenizer, prompt, inference_config)
            generation_time = time.time() - start_time
            
            # Enregistrer la réponse
            df.at[i, 'model_response'] = output_text
            df.at[i, 'generation_time'] = generation_time
            
            # Afficher la progression
            logger.info(f"Traité {i+1}/{len(df)} - Temps: {generation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la ligne {i}: {e}")
            df.at[i, 'model_response'] = f"ERREUR: {str(e)}"
    
    # Calculer le temps total
    total_time = time.time() - total_start_time
    logger.info(f"Traitement terminé en {total_time:.2f} secondes")
    logger.info(f"Temps moyen par exemple: {total_time/len(df):.2f} secondes")
    
    # Sauvegarder les résultats
    if output_file is None:
        # Créer un nom de fichier par défaut
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_results{input_path.suffix}")
    
    df.to_csv(output_file, index=False)
    logger.info(f"Résultats sauvegardés dans {output_file}")
    
    return df

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Inférence avec un modèle comptable fine-tuné")
    parser.add_argument("--model_path", type=str, default="hf_model_export", 
                        help="Chemin vers le modèle sauvegardé")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"], default="interactive",
                        help="Mode d'inférence: interactif ou par lot")
    parser.add_argument("--input_file", type=str, 
                        help="Fichier CSV d'entrée pour le mode batch")
    parser.add_argument("--output_file", type=str, 
                        help="Fichier CSV de sortie pour le mode batch")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Longueur maximale de séquence")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Utiliser la quantification 4-bit")
    parser.add_argument("--cpu", action="store_true",
                        help="Forcer l'utilisation du CPU")
    
    args = parser.parse_args()
    
    # Déterminer le périphérique
    device = "cpu" if args.cpu else None
    
    try:
        # Initialiser le modèle
        model, tokenizer, inference_config = initialize_model(
            args.model_path, 
            args.max_seq_length,
            args.load_in_4bit,
            device
        )
        
        # Exécuter le mode approprié
        if args.mode == "interactive":
            interactive_evaluation(model, tokenizer, inference_config)
        else:  # batch
            if not args.input_file:
                parser.error("--input_file est requis pour le mode batch")
            
            batch_evaluation_from_file(
                model, 
                tokenizer, 
                args.input_file, 
                args.output_file,
                inference_config
            )
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
