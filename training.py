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
from huggingface_hub import HfApi, snapshot_download

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppression d'avertissements inutiles
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
torch.cuda.empty_cache()  # Libération de la mémoire GPU

def check_dependencies():
    """Vérifie et installe les dépendances manquantes."""
    required = ["torch", "transformers", "peft", "diffusers", "trl", "datasets", "easydict", "wandb"]
    for pkg in required:
        if importlib.util.find_spec(pkg) is None:
            logger.warning(f"{pkg} n'est pas installé. Installation en cours...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            logger.info(f"{pkg} installé avec succès.")

def clone_repository(repo_url="https://github.com/Wan-Video/Wan2.1.git", clone_dir="Wan2_1"):
    """Clone le dépôt contenant le code du modèle."""
    if not os.path.exists(clone_dir):
        logger.info("Clonage du dépôt...")
        try:
            subprocess.check_call(["git", "clone", repo_url, clone_dir])
            logger.info("Dépôt cloné avec succès.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors du clonage du dépôt : {e}")
            raise e
    else:
        logger.info("Dépôt déjà présent, clonage ignoré.")
    sys.path.append(os.path.abspath(clone_dir))

def download_model_weights(model_name="Wan-AI/Wan2.1-T2V-14B", local_dir="Wan2.1-T2V-14B"):
    """Télécharge les poids du modèle depuis Hugging Face."""
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        logger.info("Téléchargement des poids du modèle...")
        snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
        logger.info("Téléchargement terminé.")
    else:
        logger.info("Répertoire existant pour les poids, téléchargement ignoré.")
    return local_dir

def load_module_from_file(file_path, module_name):
    """Charge dynamiquement un module à partir d'un fichier."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        logger.error(f"Impossible de charger le module {module_name} depuis {file_path}")
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_model_and_tokenizer():
    """Charge le modèle et le tokenizer, et configure LoRA sur l'UNet et le text encoder."""
    check_dependencies()
    clone_repository()
    model_dir = download_model_weights()

    repo_files = os.listdir("Wan2_1") if os.path.exists("Wan2_1") else []
    logger.info(f"Fichiers présents dans le dépôt : {repo_files}")

    model = None

    try:
        # Tentative via generate.py
        if "generate.py" in repo_files:
            generate_module = load_module_from_file(os.path.join("Wan2_1", "generate.py"), "generate")
            if generate_module:
                available_funcs = [f for f in dir(generate_module) if callable(getattr(generate_module, f)) and not f.startswith("_")]
                candidate_funcs = [f for f in available_funcs if "pipeline" in f.lower() or "load" in f.lower()]
                for func_name in candidate_funcs:
                    try:
                        func = getattr(generate_module, func_name)
                        model = func(
                            task="t2v-14B",
                            ckpt_dir=model_dir,
                            size="1280*720",
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                        )
                        logger.info(f"Modèle chargé avec succès via generate.py ({func_name})")
                        break
                    except Exception as e:
                        logger.warning(f"Échec avec {func_name} dans generate.py : {e}")
                        continue
        # Recherche dans d'autres fichiers Python du dépôt
        if model is None:
            python_files = [f for f in repo_files if f.endswith(".py") and f != "generate.py"]
            for py_file in python_files:
                module = load_module_from_file(os.path.join("Wan2_1", py_file), py_file[:-3])
                if module:
                    available_funcs = [f for f in dir(module) if callable(getattr(module, f)) and not f.startswith("_")]
                    candidate_funcs = [f for f in available_funcs if "pipeline" in f.lower() or "load" in f.lower()]
                    for func_name in candidate_funcs:
                        try:
                            func = getattr(module, func_name)
                            model = func(
                                task="t2v-14B",
                                ckpt_dir=model_dir,
                                size="1280*720",
                                device="cuda" if torch.cuda.is_available() else "cpu",
                                dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                            )
                            logger.info(f"Modèle chargé via {py_file} ({func_name})")
                            break
                        except Exception as e:
                            logger.warning(f"Échec avec {py_file}.{func_name} : {e}")
                            continue
                    if model is not None:
                        break
        # En dernier recours, utilisation directe de DiffusionPipeline
        if model is None:
            logger.warning("Aucune fonction de chargement trouvée. Utilisation de DiffusionPipeline directement.")
            model = DiffusionPipeline.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.info("Modèle chargé via DiffusionPipeline")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        raise e

    if not hasattr(model, "unet") or model.unet is None:
        raise ValueError("L'UNet n'a pas été chargé correctement.")

    # Application de LoRA sur l'UNet
    lora_config_unet = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none"
    )
    model.unet = get_peft_model(model.unet, lora_config_unet)
    logger.info("LoRA appliqué à l'UNet")

    # Application de LoRA sur le text encoder, si présent
    if hasattr(model, "text_encoder") and model.text_encoder is not None:
        lora_config_text = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["q", "k", "v"],
            lora_dropout=0.05,
            bias="none"
        )
        model.text_encoder = get_peft_model(model.text_encoder, lora_config_text)
        logger.info("LoRA appliqué au text encoder")

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        logger.info("Fallback sur le tokenizer t5-base")
    return model, tokenizer

def prepare_dataset(tokenizer, data_dir="dataset"):
    """Prépare le dataset en associant images et textes, et applique le prétraitement."""
    preprocess = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_text_pairs = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(data_dir, file)
            base_name = os.path.splitext(file)[0]
            text_file = os.path.join(data_dir, f"{base_name}.txt")
            if os.path.exists(text_file):
                with open(text_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            else:
                text = base_name.replace("_", " ")
            image_text_pairs.append({"image_path": image_path, "text": text + " TOK"})
    dataset = Dataset.from_dict({
        "image_path": [item["image_path"] for item in image_text_pairs],
        "text": [item["text"] for item in image_text_pairs]
    })
    def load_and_preprocess(example):
        image = Image.open(example["image_path"]).convert("RGB")
        example["image"] = preprocess(image)
        return example
    dataset = dataset.map(load_and_preprocess)
    logger.info(f"Dataset préparé avec {len(dataset)} échantillons")
    return dataset

def data_collator(examples, tokenizer):
    """Collateur pour préparer les batchs d'entraînement."""
    texts = [ex["text"] for ex in examples]
    text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    images = torch.stack([ex["image"] for ex in examples])
    text_inputs["images"] = images
    text_inputs["labels"] = images  # Les images sont les labels pour la diffusion
    return text_inputs

def setup_trainer(model, tokenizer, dataset):
    """Configure l'entraîneur avec les hyperparamètres et l'intégration W&B."""
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
        model=model.unet,  # On entraîne l'UNet avec LoRA
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=lambda examples: data_collator(examples, tokenizer)
    )
    return trainer

def check_disk_space(required_gb=10):
    """Vérifie que l'espace disque disponible est suffisant."""
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    if free_gb < required_gb:
        raise RuntimeError(f"Espace disque insuffisant : {free_gb:.2f}GB disponibles, {required_gb}GB requis.")

def export_model(model, tokenizer, export_dir="hf_model_export"):
    """Exporte le modèle finetuné et le tokenizer, avec sauvegarde des poids LoRA."""
    check_disk_space(required_gb=10)
    os.makedirs(export_dir, exist_ok=True)
    abs_export_dir = os.path.abspath(export_dir)
    logger.info(f"Exportation du modèle dans {abs_export_dir}")
    
    if hasattr(model, "unet"):
        model.unet.save_pretrained(os.path.join(export_dir, "unet_lora"))
    if hasattr(model, "text_encoder"):
        model.text_encoder.save_pretrained(os.path.join(export_dir, "text_encoder_lora"))
    if tokenizer:
        tokenizer.save_pretrained(export_dir)
    
    # Création d'un README explicatif
    readme_path = os.path.join(export_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Modèle Wan2.1 Fine-Tuné avec LoRA\n\n")
        f.write("Ce dossier contient les poids LoRA pour l'UNet et le text encoder.\n")
        f.write("Pour charger le modèle avec LoRA, utilisez :\n")
        f.write("```python\n")
        f.write("from peft import PeftModel\n")
        f.write("base_model = ...  # Charger le modèle de base Wan2.1-T2V-14B\n")
        f.write("model.unet = PeftModel.from_pretrained(base_model.unet, 'unet_lora')\n")
        f.write("model.text_encoder = PeftModel.from_pretrained(base_model.text_encoder, 'text_encoder_lora')\n")
        f.write("```\n")
    logger.info("Modèle exporté avec succès.")

def upload_to_hf(export_dir, repo_id):
    """Téléverse le modèle exporté sur Hugging Face."""
    api = HfApi()
    api.upload_folder(
        folder_path=export_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    logger.info(f"Modèle téléversé sur {repo_id}")

def main():
    try:
        check_dependencies()
        model, tokenizer = load_model_and_tokenizer()
        dataset = prepare_dataset(tokenizer, data_dir="dataset")
        trainer = setup_trainer(model, tokenizer, dataset)
        
        # Initialisation de Weights & Biases
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
        
        # Option de téléversement sur Hugging Face
        upload = input("Voulez-vous téléverser le modèle sur Hugging Face ? (yes/no): ").strip().lower()
        if upload in ["yes", "y", "oui", "o"]:
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
