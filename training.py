import torch
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from diffusers import DiffusionPipeline
from trl import SFTTrainer
import os
import logging
from datasets import load_dataset
from huggingface_hub import HfApi

# Configuration du logger pour suivre l'exécution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Libérer la mémoire GPU avant de commencer
torch.cuda.empty_cache()

def load_model_and_tokenizer():
    """Charge le modèle Wan2.1-T2V-14B et le tokenizer, et configure LoRA."""
    logger.info("Initialisation du modèle Wan2.1-T2V-14B...")
    
    # Charger le modèle depuis Hugging Face
    model = DiffusionPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-14B",  # Identifiant du modèle (à vérifier)
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger le tokenizer associé
    tokenizer = AutoTokenizer.from_pretrained("Wan-AI/Wan2.1-T2V-14B")
    
    # Configuration de LoRA pour le fine-tuning
    lora_config = LoraConfig(
        r=32,               # Rang de LoRA
        lora_alpha=32,      # Facteur d'échelle
        target_modules=["unet", "vae"],  # Modules ciblés (à adapter si nécessaire)
        lora_dropout=0.05,  # Taux de dropout
        bias="none"         # Pas de biais
    )
    
    # Appliquer LoRA au sous-modèle UNET
    model.unet = get_peft_model(model.unet, lora_config)
    
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
        model=model.unet,                # Entraîner uniquement le UNET
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
    model.save_pretrained(export_dir, max_shard_size="5GB")
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
    export_model(model.unet, tokenizer, export_dir)
    
    # Téléverser sur Hugging Face
    repo_id = "votre-nom-utilisateur/nom-du-modele"  # À personnaliser
    upload_to_hf(export_dir, repo_id)
    
    logger.info("Processus terminé avec succès.")

if __name__ == "__main__":
    main()
