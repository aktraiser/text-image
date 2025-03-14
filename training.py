import os
import sys
import torch
import logging
import warnings
import subprocess
import importlib.util
import shutil
from PIL import Image
from torchvision import transforms
import wandb

from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from diffusers import DiffusionPipeline
from trl import SFTTrainer
from datasets import Dataset
from huggingface_hub import HfApi

# Configuration du logger et gestion des avertissements
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
torch.cuda.empty_cache()

def check_dependencies():
    """Vérifie et installe les dépendances manquantes."""
    required = ["torch", "transformers", "peft", "diffusers", "trl", "datasets", "wandb"]
    for pkg in required:
        if importlib.util.find_spec(pkg) is None:
            logger.warning(f"{pkg} n'est pas installé. Installation en cours...")
            subprocess.run(["pip", "install", pkg], check=True)
            logger.info(f"{pkg} installé avec succès.")

def load_model_and_tokenizer():
    """Charge directement le modèle depuis Hugging Face et applique LoRA."""
    check_dependencies()
    
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "balanced" if torch.cuda.is_available() else None

    try:
        # Chargement direct depuis Hugging Face
        model = DiffusionPipeline.from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B",
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        logger.info("Modèle chargé directement depuis Hugging Face.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle depuis Hugging Face : {e}")
        raise e

    if not hasattr(model, "unet") or model.unet is None:
        raise ValueError("L'UNet n'a pas été chargé correctement.")

    # Configuration et application de LoRA sur l'UNet
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none"
    )
    model.unet = get_peft_model(model.unet, lora_config)
    logger.info("LoRA appliqué à l'UNet")

    # Si le modèle contient un text encoder, on applique LoRA dessus
    if hasattr(model, "text_encoder") and model.text_encoder is not None:
        text_lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["q", "k", "v"],
            lora_dropout=0.05,
            bias="none"
        )
        model.text_encoder = get_peft_model(model.text_encoder, text_lora_config)
        logger.info("LoRA appliqué au text encoder")

    # Chargement du tokenizer (si présent dans le pipeline, sinon fallback)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        logger.info("Fallback sur le tokenizer t5-base")
    
    return model, tokenizer

def prepare_dataset(tokenizer, data_path="dataset"):
    """Prépare le dataset en associant images et textes avec prétraitement."""
    logger.info(f"Chargement des données depuis : {data_path}")
    preprocess = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_text_pairs = []
    for file in os.listdir(data_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(data_path, file)
            base_name = os.path.splitext(file)[0]
            text_path = os.path.join(data_path, f"{base_name}.txt")
            text = open(text_path, 'r', encoding='utf-8').read().strip() if os.path.exists(text_path) else base_name.replace('_', ' ')
            image_text_pairs.append({"image_path": image_path, "text": f"{text} TOK"})
    
    dataset = Dataset.from_dict({
        "image_path": [item["image_path"] for item in image_text_pairs],
        "text": [item["text"] for item in image_text_pairs]
    })
    
    def load_and_preprocess(example):
        image = Image.open(example["image_path"]).convert("RGB")
        example["image"] = preprocess(image)
        return example
    
    dataset = dataset.map(load_and_preprocess)
    logger.info(f"Dataset prêt avec {len(dataset)} paires image-texte")
    return dataset

def wan_data_collator(examples, tokenizer):
    """Collateur adapté aux modèles de diffusion."""
    texts = [ex["text"] for ex in examples]
    text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    images = torch.stack([ex["image"] for ex in examples])
    text_inputs["images"] = images
    text_inputs["labels"] = images  # Les images servent de labels
    return text_inputs

def setup_trainer(model, tokenizer, dataset):
    """Configure l'entraîneur et l'intégration W&B."""
    os.environ["WANDB_PROJECT"] = "wan_finetune"
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=10,
        learning_rate=1e-4,
        optim="adamw_8bit",
        logging_steps=5,
        save_steps=10,
        gradient_checkpointing=False,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        logging_dir="./logs",
        save_strategy="steps",
        save_total_limit=3,
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model.unet,  # Entraînement de l'UNet avec LoRA
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=lambda examples: wan_data_collator(examples, tokenizer)
    )
    return trainer

def check_disk_space(required_space_gb=10):
    """Vérifie l'espace disque disponible."""
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    if free_gb < required_space_gb:
        raise RuntimeError(f"Espace disque insuffisant : {free_gb:.2f} GB disponibles, {required_space_gb} GB requis.")

def export_model(model, tokenizer, export_dir="hf_model_export"):
    """Exporte le modèle finetuné et le tokenizer."""
    check_disk_space(required_space_gb=10)
    os.makedirs(export_dir, exist_ok=True)
    abs_export_dir = os.path.abspath(export_dir)
    logger.info(f"Exportation du modèle dans {abs_export_dir}")
    
    if hasattr(model, "unet"):
        model.unet.save_pretrained(os.path.join(export_dir, "unet_lora"))
    if hasattr(model, "text_encoder"):
        model.text_encoder.save_pretrained(os.path.join(export_dir, "text_encoder_lora"))
    if tokenizer:
        tokenizer.save_pretrained(export_dir)
    
    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Modèle Wan2.1 Fine-Tuné avec LoRA\n\n")
        f.write("Ce dossier contient les poids LoRA pour l'UNet et le text encoder.\n")
        f.write("Pour charger le modèle avec LoRA :\n")
        f.write("```python\n")
        f.write("from peft import PeftModel\n")
        f.write("base_model = ...  # Charger Wan2.1-T2V-14B\n")
        f.write("model.unet = PeftModel.from_pretrained(base_model.unet, 'unet_lora')\n")
        f.write("model.text_encoder = PeftModel.from_pretrained(base_model.text_encoder, 'text_encoder_lora')\n")
        f.write("```\n")
    logger.info("Modèle exporté avec succès.")

def upload_to_hf(export_dir, repo_id):
    """Téléverse le modèle sur Hugging Face."""
    api = HfApi()
    api.upload_folder(folder_path=export_dir, repo_id=repo_id, repo_type="model")
    logger.info(f"Modèle téléversé sur {repo_id}")

def main():
    try:
        model, tokenizer = load_model_and_tokenizer()
        dataset = prepare_dataset(tokenizer, "dataset")
        trainer = setup_trainer(model, tokenizer, dataset)
        
        wandb.init(
            entity="votre_entite",  # Remplacez par votre entité W&B
            project="wan_finetune",
            config={
                "learning_rate": trainer.args.learning_rate,
                "architecture": "Wan2.1-T2V-14B avec LoRA",
                "dataset": "custom_dataset",
                "epochs": trainer.args.num_train_epochs,
            }
        )
        
        logger.info("Début de l'entraînement...")
        trainer.train()
        
        if trainer.args.output_dir:
            logger.info("Sauvegarde du modèle final...")
            trainer.save_model()
        
        export_model(model, tokenizer, export_dir="hf_model_export")
        
        upload_choice = input("Voulez-vous téléverser le modèle sur Hugging Face ? (yes/no): ").strip().lower()
        if upload_choice in ["yes", "y", "oui", "o"]:
            repo_id = input("Entrez le nom du dépôt (format username/model-name): ").strip()
            upload_to_hf("hf_model_export", repo_id)
        else:
            logger.info("Téléversement ignoré, modèle sauvegardé localement.")
    except Exception as e:
        logger.error(f"Erreur dans le processus principal : {e}")
    finally:
        wandb.finish()
        logger.info("Processus terminé.")

if __name__ == "__main__":
    main()
