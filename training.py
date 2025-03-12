import torch
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from diffusers import DiffusionPipeline  # Hypothèse : Wan2.1-T2V-14B utilise une pipeline de diffusion
from trl import SFTTrainer
import os
import logging
from datasets import load_dataset

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nettoyer la mémoire GPU
torch.cuda.empty_cache()

def initialize_model():
    """Charge le modèle Wan2.1-T2V-14B et configure LoRA"""
    logger.info("Chargement du modèle Wan2.1-T2V-14B...")
    
    # Charger le modèle (remplacez par le chemin exact ou l'identifiant du modèle)
    model = DiffusionPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-14B",  # À adapter selon l'identifiant réel
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger le tokenizer (si applicable)
    tokenizer = AutoTokenizer.from_pretrained("Wan-AI/Wan2.1-T2V-14B")
    
    # Configuration LoRA (basée sur les hyperparamètres Replicate)
    lora_config = LoraConfig(
        r=32,  # lora_rank
        lora_alpha=32,
        target_modules=["unet", "vae"],  # À adapter selon l'architecture réelle du modèle
        lora_dropout=0.05,  # caption_dropout_rate
        bias="none"
    )
    
    # Appliquer LoRA au modèle (par exemple au composant UNET si c'est un modèle de diffusion)
    model.unet = get_peft_model(model.unet, lora_config)
    
    return model, tokenizer

def initialize_dataset(tokenizer, input_images_path):
    """Charge et prépare le jeu de données texte-vidéo"""
    logger.info(f"Chargement du jeu de données depuis : {input_images_path}")
    
    # Charger un dataset local (par exemple, paires texte-images ou texte-vidéos)
    dataset = load_dataset("imagefolder", data_dir=input_images_path, split="train")
    
    def preprocess(example):
        """Formate les exemples pour l'entraînement"""
        # Ajouter un trigger word si nécessaire
        text = f"{example['text']} TOK"  # trigger_word = "TOK"
        return {"text": text}
    
    dataset = dataset.map(preprocess)
    return dataset

def initialize_trainer(model, tokenizer, dataset):
    """Configure l'entraîneur avec les hyperparamètres de Replicate"""
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=1,  # batch_size
        num_train_epochs=1,  # Steps = 2000, à ajuster selon la taille du dataset
        max_steps=2000,  # steps
        learning_rate=0.0001,  # learning_rate
        optim="adamw_8bit",  # optimizer
        logging_steps=250,  # wandb_save_interval
        save_steps=250,
        gradient_checkpointing=False,  # gradient_checkpointing
        fp16=torch.cuda.is_available(),  # Utiliser FP16 si GPU disponible
        report_to="wandb",  # Intégration Weights & Biases
        wandb_project="wan_train_replicate",  # wandb_project
        logging_dir="./logs",
        save_strategy="steps",
        save_total_limit=3,
    )
    
    trainer = SFTTrainer(
        model=model.unet,  # Entraîner uniquement la partie UNET avec LoRA
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=training_args,
    )
    
    return trainer

def save_model(model, tokenizer, output_dir):
    """Sauvegarde le modèle fine-tuné"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Sauvegarde du modèle dans : {output_dir}")
    
    # Sauvegarder le modèle avec LoRA
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def main():
    # Initialiser le modèle et le tokenizer
    model, tokenizer = initialize_model()
    
    # Charger le jeu de données (remplacez par le chemin réel)
    dataset = initialize_dataset(tokenizer, "path/to/your/input_images")
    
    # Initialiser le trainer
    trainer = initialize_trainer(model, tokenizer, dataset)
    
    # Lancer l'entraînement
    logger.info("Début de l'entraînement...")
    trainer.train()
    
    # Sauvegarder le modèle
    save_model(model.unet, tokenizer, "hf_model_export")
    logger.info("Entraînement terminé et modèle sauvegardé.")

if __name__ == "__main__":
    main()
