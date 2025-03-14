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
from PIL import Image
import shutil
import glob
import json
from torchvision import transforms
import wandb  # Importation de Weights & Biases

# Configuration du logger pour suivre l'exécution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Libérer la mémoire GPU avant de commencer
torch.cuda.empty_cache()

# Vérification et installation des dépendances
def check_dependencies():
    """Vérifie et installe les dépendances manquantes, y compris W&B."""
    required = ["torch", "transformers", "peft", "diffusers", "trl", "datasets", "easydict", "wandb"]
    for pkg in required:
        if not importlib.util.find_spec(pkg):
            logger.warning(f"{pkg} n'est pas installé. Installation en cours...")
            subprocess.run(["pip", "install", pkg], check=True)
            logger.info(f"{pkg} installé avec succès.")

# Clonage du dépôt Wan2.1
def clone_wan_repository():
    """Clone le dépôt Wan2.1 pour accéder au code du modèle."""
    logger.info("Clonage du dépôt Wan2.1...")
    if not os.path.exists("Wan2_1"):
        subprocess.run(["git", "clone", "https://github.com/Wan-Video/Wan2.1.git", "Wan2_1"], check=True)
        logger.info("Dépôt cloné avec succès.")
    else:
        logger.info("Dépôt déjà existant, clonage ignoré.")
    sys.path.append(os.path.abspath("Wan2_1"))

# Chargement dynamique d’un module
def load_module_from_file(file_path, module_name):
    """Charge un module Python à partir d’un chemin de fichier."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Téléchargement des poids du modèle
def download_model_weights(model_name="Wan-AI/Wan2.1-T2V-14B", local_dir="./Wan2.1-T2V-14B"):
    """Télécharge les poids du modèle depuis Hugging Face."""
    logger.info(f"Téléchargement des poids du modèle pour {model_name}...")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
        logger.info(f"Poids du modèle téléchargés dans {local_dir}")
    else:
        logger.info(f"Le répertoire des poids {local_dir} existe déjà, téléchargement ignoré.")
    return local_dir

# Chargement du modèle et du tokenizer
def load_model_and_tokenizer():
    """Charge le modèle Wan2.1-T2V-14B et le tokenizer, et configure LoRA."""
    logger.info("Initialisation du modèle Wan2.1-T2V-14B...")
    
    # Vérifier les dépendances
    check_dependencies()
    
    # Cloner le dépôt et télécharger les poids
    clone_wan_repository()
    model_dir = download_model_weights()
    
    # Importer le module personnalisé depuis le dépôt cloné
    try:
        generate_module = load_module_from_file(os.path.join("Wan2_1", "generate.py"), "generate")
        if generate_module and hasattr(generate_module, "load_t2v_pipeline"):
            model = generate_module.load_t2v_pipeline(
                task="t2v-14B",
                ckpt_dir=model_dir,
                size="1280*720",
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            logger.info("Modèle chargé avec succès via le pipeline personnalisé")
        else:
            raise ImportError("Fonction load_t2v_pipeline introuvable")
    except Exception as e:
        logger.error(f"Échec du chargement du modèle : {str(e)}")
        raise RuntimeError("Impossible de charger le modèle Wan2.1-T2V-14B.")

    # Vérification de l’UNet
    if not hasattr(model, "unet") or model.unet is None:
        raise ValueError("L’UNet n’a pas été chargé correctement.")

    # Configuration de LoRA pour l’UNet
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none"
    )
    model.unet = get_peft_model(model.unet, lora_config)
    logger.info("LoRA appliqué à l’UNet")

    # Configuration de LoRA pour le text encoder (si présent)
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

    # Chargement du tokenizer
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        logger.info("Utilisation du tokenizer t5-base comme fallback")

    return model, tokenizer

# Préparation du jeu de données
def prepare_dataset(tokenizer, data_path="dataset"):
    """Charge et prépare le jeu de données avec prétraitement des images."""
    logger.info(f"Chargement des données depuis : {data_path}")
    
    # Préprocesseur d’images
    preprocess = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Appariement images-textes
    image_text_pairs = []
    for file in os.listdir(data_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(data_path, file)
            base_name = os.path.splitext(file)[0]
            text_path = os.path.join(data_path, f"{base_name}.txt")
            text = open(text_path, 'r', encoding='utf-8').read().strip() if os.path.exists(text_path) else base_name.replace('_', ' ')
            image_text_pairs.append({"image_path": image_path, "text": f"{text} TOK"})
    
    # Création du dataset
    from datasets import Dataset
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

# Collateur de données
def wan_data_collator(examples):
    """Collateur adapté aux modèles de diffusion."""
    batch = {}
    texts = [example["text"] for example in examples]
    text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    batch.update(text_inputs)
    images = torch.stack([example["image"] for example in examples])
    batch["images"] = images
    batch["labels"] = images  # Les images servent de labels pour la diffusion
    return batch

# Configuration de l’entraîneur
def setup_trainer(model, tokenizer, dataset):
    """Configure l’entraîneur avec les hyperparamètres et W&B."""
    os.environ["WANDB_PROJECT"] = "wan_finetune"
    
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=10,
        learning_rate=0.0001,
        optim="adamw_8bit",
        logging_steps=5,
        save_steps=10,
        gradient_checkpointing=False,
        fp16=torch.cuda.is_available(),
        report_to="wandb",  # Intégration avec W&B
        logging_dir="./logs",
        save_strategy="steps",
        save_total_limit=3,
        remove_unused_columns=False,
    )
    
    trainer = SFTTrainer(
        model=model.unet,  # Entraîner l’UNet avec LoRA
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=wan_data_collator,
    )
    return trainer

# Vérification de l’espace disque
def check_disk_space(required_space_gb=10):
    """Vérifie l’espace disque disponible."""
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    if free_gb < required_space_gb:
        raise RuntimeError(f"Espace disque insuffisant : {free_gb:.2f} GB disponibles, {required_space_gb} GB requis.")

# Exportation du modèle
def export_model(model, tokenizer, export_dir, training_args=None):
    """Exporte le modèle avec vérification de l’espace disque."""
    check_disk_space(required_space_gb=10)
    os.makedirs(export_dir, exist_ok=True)
    abs_output_dir = os.path.abspath(export_dir)
    logger.info(f"Exportation du modèle vers : {abs_output_dir}")
    
    # Sauvegarde des poids LoRA
    if hasattr(model, "unet"):
        model.unet.save_pretrained(os.path.join(export_dir, "unet_lora"))
    if hasattr(model, "text_encoder"):
        model.text_encoder.save_pretrained(os.path.join(export_dir, "text_encoder_lora"))
    
    # Sauvegarde du tokenizer
    if tokenizer:
        tokenizer.save_pretrained(export_dir)
    
    # Ajout d’un README
    with open(os.path.join(export_dir, "README.md"), "w") as f:
        f.write("# Modèle Wan2.1 Fine-Tuné avec LoRA\n\n")
        f.write("Ce dossier contient les poids LoRA pour l’UNet et le text encoder.\n")
        f.write("Pour charger le modèle avec LoRA :\n")
        f.write("```python\n")
        f.write("from peft import PeftModel\n")
        f.write("base_model = ...  # Charger Wan2.1-T2V-14B\n")
        f.write("model.unet = PeftModel.from_pretrained(base_model.unet, 'unet_lora')\n")
        f.write("model.text_encoder = PeftModel.from_pretrained(base_model.text_encoder, 'text_encoder_lora')\n")
        f.write("```\n")
    logger.info("Modèle exporté avec succès.")

# Fonction principale
def main():
    """Orchestre l’entraînement et l’exportation avec suivi W&B."""
    check_dependencies()
    
    # Charger le modèle et le tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Préparer le dataset
    dataset = prepare_dataset(tokenizer, "dataset")
    
    # Configurer l’entraîneur
    trainer = setup_trainer(model, tokenizer, dataset)
    
    # Initialisation de Weights & Biases
    run = wandb.init(
        entity="votre_entite",  # Remplacez par votre entité W&B
        project="wan_finetune",
        config={
            "learning_rate": trainer.args.learning_rate,
            "architecture": "Wan2.1-T2V-14B avec LoRA",
            "dataset": "custom_dataset",
            "epochs": trainer.args.num_train_epochs,
        }
    )
    
    # Lancer l’entraînement
    logger.info("Lancement de l’entraînement...")
    trainer.train()
    
    # Sauvegarder le modèle final
    if trainer.args.output_dir:
        logger.info("Sauvegarde du modèle final...")
        trainer.save_model()
    
    # Exporter le modèle localement
    export_dir = "hf_model_export"
    export_model(model, tokenizer, export_dir, trainer.args)
    
    # Option de téléversement sur Hugging Face
    upload_choice = input("Voulez-vous téléverser le modèle sur Hugging Face ? (yes/no): ")
    if upload_choice.lower() in ["yes", "y", "oui", "o"]:
        repo_id = input("Entrez le nom du dépôt (format username/model-name): ")
        upload_to_hf(export_dir, repo_id)
    else:
        logger.info("Téléversement ignoré. Modèle sauvegardé localement.")
    
    # Terminer la session W&B
    run.finish()
    
    logger.info("Processus terminé avec succès.")

# Fonction placeholder pour le téléversement (à implémenter si nécessaire)
def upload_to_hf(export_dir, repo_id):
    """Téléverse le modèle sur Hugging Face."""
    api = HfApi()
    api.upload_folder(
        folder_path=export_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    logger.info(f"Modèle téléversé avec succès sur {repo_id}")

if __name__ == "__main__":
    main()
