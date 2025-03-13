import torch
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from diffusers import DiffusionPipeline
from trl import SFTTrainer
import os
import logging
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
import sys
import subprocess
import importlib.util

# Configuration du logger pour suivre l'exécution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def load_model_and_tokenizer():
    """Charge le modèle Wan2.1-T2V-14B et le tokenizer, et configure LoRA."""
    logger.info("Initialisation du modèle Wan2.1-T2V-14B...")
    
    # Clone the repository and download weights
    clone_wan_repository()
    model_dir = download_model_weights()
    
    # Import the custom modules from the cloned repository
    try:
        # Try to load the generate.py module directly using importlib
        generate_module = load_module_from_file(os.path.join("Wan2_1", "generate.py"), "generate")
        
        if generate_module and hasattr(generate_module, "load_t2v_pipeline"):
            # Load the model using the custom function
            model = generate_module.load_t2v_pipeline(
                task="t2v-14B",
                ckpt_dir=model_dir,
                size="1280*720",  # Default resolution
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            logger.info("Successfully loaded model using custom pipeline")
        else:
            raise ImportError("Could not find load_t2v_pipeline function")
            
    except (ImportError, FileNotFoundError) as e:
        # Fallback to using a direct approach
        logger.info(f"Using fallback method to load the model: {str(e)}")
        
        # Create a simple wrapper around the model
        class WanModelWrapper:
            def __init__(self, model_dir):
                self.model_dir = model_dir
                self.unet = None  # Will be populated later
                self.tokenizer = None
                
                # Try to load the UNet component directly
                try:
                    from diffusers import UNet2DConditionModel
                    self.unet = UNet2DConditionModel.from_pretrained(
                        os.path.join(model_dir, "unet"),
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    logger.info("Successfully loaded UNet component")
                except Exception as unet_error:
                    logger.warning(f"Could not load UNet: {str(unet_error)}")
                
                # Load the tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
                    logger.info("Successfully loaded tokenizer")
                except Exception:
                    logger.warning("Could not load tokenizer, using t5-base instead")
                    self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        
        model = WanModelWrapper(model_dir)
    
    # Configure LoRA for fine-tuning
    lora_config = LoraConfig(
        r=32,               # Rang de LoRA
        lora_alpha=32,      # Facteur d'échelle
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # Modules ciblés pour les transformers
        lora_dropout=0.05,  # Taux de dropout
        bias="none"         # Pas de biais
    )
    
    # Extract the UNet from the pipeline if it's a DiffusionPipeline
    if hasattr(model, "unet") and model.unet is not None:
        try:
            model.unet = get_peft_model(model.unet, lora_config)
            logger.info("Successfully applied LoRA to UNet")
        except Exception as lora_error:
            logger.error(f"Failed to apply LoRA: {str(lora_error)}")
    else:
        logger.warning("Could not apply LoRA to UNet - model structure is different than expected")
    
    # Use the tokenizer from the model if available, otherwise use T5
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        logger.info("Using t5-base tokenizer as fallback")
    
    return model, tokenizer

def prepare_dataset(tokenizer, data_path="dataset"):
    """Charge et prépare le jeu de données à partir d'un dossier local."""
    logger.info(f"Chargement des données depuis : {data_path}")
    
    # Charger le dataset depuis un dossier d'images
    dataset = load_dataset("imagefolder", data_dir=data_path, split="train")
    logger.info(f"Dataset chargé avec {len(dataset)} exemples.")
    
    def preprocess_data(example):
        """Ajoute un mot-clé déclencheur au texte."""
        text = f"{example['text']} TOK"  # "TOK" comme mot déclencheur
        return {"text": text}
    
    # Préparer les données
    dataset = dataset.map(preprocess_data)
    return dataset

def setup_trainer(model, tokenizer, dataset):
    """Configure l'entraîneur avec les hyperparamètres d'entraînement."""
    training_args = TrainingArguments(
        output_dir="./outputs",          # Dossier de sortie
        per_device_train_batch_size=1,   # Taille du batch par appareil
        num_train_epochs=1,              # Nombre d'époques
        max_steps=2000,                  # Nombre total de pas
        learning_rate=0.0001,            # Taux d'apprentissage
        optim="adamw_8bit",              # Optimiseur adapté
        logging_steps=250,               # Intervalle de logs
        save_steps=250,                  # Intervalle de sauvegarde
        gradient_checkpointing=False,    # Pas de checkpointing
        fp16=torch.cuda.is_available(),  # Utiliser FP16 si GPU disponible
        report_to="wandb",               # Suivi avec Weights & Biases
        wandb_project="wan_finetune",    # Nom du projet W&B
        logging_dir="./logs",            # Dossier des logs
        save_strategy="steps",           # Sauvegarde par étapes
        save_total_limit=3,              # Limite de sauvegardes
    )
    
    # Initialiser l'entraîneur
    trainer = SFTTrainer(
        model=model.unet if hasattr(model, "unet") else model,  # Entraîner uniquement le UNET
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",       # Champ texte du dataset
        args=training_args,
    )
    
    return trainer

def export_model(model, tokenizer, export_dir):
    """Sauvegarde le modèle fine-tuné localement."""
    os.makedirs(export_dir, exist_ok=True)
    logger.info(f"Exportation du modèle vers : {export_dir}")
    
    # Sauvegarder le modèle avec sharding
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(export_dir, max_shard_size="5GB")
    else:
        # Fallback for custom model structure
        if hasattr(model, "unet") and hasattr(model.unet, "save_pretrained"):
            model.unet.save_pretrained(os.path.join(export_dir, "unet"), max_shard_size="5GB")
    
    # Save tokenizer if available
    if tokenizer is not None:
        tokenizer.save_pretrained(export_dir)
    
    # Ajouter un fichier .gitattributes pour Git LFS
    with open(os.path.join(export_dir, ".gitattributes"), "w") as f:
        f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.pth filter=lfs diff=lfs merge=lfs -text\n")
    
    # Ajouter un fichier README.md
    with open(os.path.join(export_dir, "README.md"), "w") as f:
        f.write("# Modèle Fine-Tuné Wan2.1-T2V-14B\n\n")
        f.write("Ce modèle est une version fine-tunée de Wan2.1-T2V-14B.\n\n")
        f.write("## Chargement\n```python\n")
        f.write("from diffusers import DiffusionPipeline\n")
        f.write('model = DiffusionPipeline.from_pretrained("votre-nom/nom-du-modele")\n')
        f.write("```\n\n")
        f.write("## Paramètres\n- Steps : 2000\n- Learning rate : 0.0001\n- Optimiseur : adamw_8bit\n")

def upload_to_hf(export_dir, repo_id):
    """Téléverse le modèle sur Hugging Face."""
    api = HfApi()
    api.upload_folder(
        folder_path=export_dir,
        repo_id=repo_id,
        repo_type="model"
    )
    logger.info(f"Modèle téléversé sur : https://huggingface.co/{repo_id}")

def main():
    """Orchestre l'entraînement, la sauvegarde et le téléversement."""
    # Charger le modèle et le tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Préparer le jeu de données
    dataset = prepare_dataset(tokenizer, "dataset")
    
    # Configurer l'entraîneur
    trainer = setup_trainer(model, tokenizer, dataset)
    
    # Lancer l'entraînement
    logger.info("Lancement de l'entraînement...")
    trainer.train()
    
    # Exporter le modèle localement
    export_dir = "hf_model_export"
    export_model(model, tokenizer, export_dir)
    
    # Téléverser sur Hugging Face
    repo_id = "votre-nom-utilisateur/nom-du-modele"  # À personnaliser
    upload_to_hf(export_dir, repo_id)
    
    logger.info("Processus terminé avec succès.")

if __name__ == "__main__":
    main()
