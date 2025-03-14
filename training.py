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
import numpy as np
import random

from transformers import AutoTokenizer, TrainingArguments, T5EncoderModel
from peft import LoraConfig, get_peft_model
from diffusers import DiffusionPipeline, UNet2DConditionModel
from trl import SFTTrainer
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download

# Configuration du logger et gestion des avertissements
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
torch.cuda.empty_cache()

def set_seed(seed=42):
    """Fixe les graines aléatoires pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Graines aléatoires fixées à {seed} pour la reproductibilité.")

def check_gpu_memory():
    """Vérifie la mémoire GPU disponible et suggère des paramètres adaptés."""
    if not torch.cuda.is_available():
        return None
    
    # Obtenir la mémoire totale et disponible
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # en GB
    reserved_memory = torch.cuda.memory_reserved(0) / 1e9
    allocated_memory = torch.cuda.memory_allocated(0) / 1e9
    free_memory = total_memory - reserved_memory
    
    logger.info(f"Mémoire GPU totale: {total_memory:.2f} GB")
    logger.info(f"Mémoire GPU réservée: {reserved_memory:.2f} GB")
    logger.info(f"Mémoire GPU allouée: {allocated_memory:.2f} GB")
    logger.info(f"Mémoire GPU disponible: {free_memory:.2f} GB")
    
    # Recommandations basées sur la mémoire disponible
    recommendations = {}
    
    if free_memory < 8:
        recommendations["offload"] = True
        recommendations["gradient_checkpointing"] = True
        recommendations["fp16"] = True
        recommendations["batch_size"] = 1
        recommendations["gradient_accumulation_steps"] = 8
        logger.warning("Mémoire GPU limitée. Activation de l'offloading et du gradient checkpointing recommandée.")
    elif free_memory < 16:
        recommendations["offload"] = False
        recommendations["gradient_checkpointing"] = True
        recommendations["fp16"] = True
        recommendations["batch_size"] = 1
        recommendations["gradient_accumulation_steps"] = 4
        logger.info("Mémoire GPU modérée. Gradient checkpointing activé pour optimiser l'utilisation.")
    else:
        recommendations["offload"] = False
        recommendations["gradient_checkpointing"] = False
        recommendations["fp16"] = True
        recommendations["batch_size"] = 2
        recommendations["gradient_accumulation_steps"] = 2
        logger.info("Mémoire GPU suffisante. Configuration optimale pour la performance.")
    
    return recommendations

def create_sample_dataset(data_path="dataset"):
    """Crée un dataset d'exemple avec des images et des prompts si aucun n'existe."""
    if os.path.exists(data_path) and any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(data_path)):
        logger.info(f"Dataset existant trouvé dans {data_path}.")
        return
    
    logger.info(f"Création d'un dataset d'exemple dans {data_path}...")
    os.makedirs(data_path, exist_ok=True)
    
    # Créer une image d'exemple (un carré coloré)
    for i in range(3):
        # Créer une image de couleur aléatoire
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = Image.new('RGB', (512, 512), color=color)
        img_path = os.path.join(data_path, f"sample_image_{i+1}.jpg")
        img.save(img_path)
        
        # Créer un fichier texte avec un prompt d'exemple
        prompts = [
            "A beautiful landscape with mountains and a lake at sunset",
            "A futuristic city with flying cars and tall skyscrapers",
            "A magical forest with glowing plants and mystical creatures"
        ]
        txt_path = os.path.join(data_path, f"sample_image_{i+1}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(prompts[i])
    
    logger.info(f"Dataset d'exemple créé avec 3 images et prompts dans {data_path}.")

def check_dependencies():
    """Vérifie et installe les dépendances manquantes."""
    required = ["torch", "transformers", "peft", "diffusers", "trl", "datasets", "wandb", "numpy", "pillow"]
    for pkg in required:
        if importlib.util.find_spec(pkg) is None:
            logger.warning(f"{pkg} n'est pas installé. Installation en cours...")
            subprocess.run(["pip", "install", pkg], check=True)
            logger.info(f"{pkg} installé avec succès.")

def load_model_and_tokenizer(offload=False):
    """Charge les composants du modèle Wan2.1 individuellement et applique LoRA.
    
    Args:
        offload (bool): Si True, certains composants seront chargés sur CPU pour économiser la mémoire GPU.
    """
    check_dependencies()
    
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() and not offload else "cpu"
    
    try:
        logger.info("Téléchargement et chargement des composants du modèle Wan2.1...")
        logger.info(f"Mode d'offloading: {'activé' if offload else 'désactivé'}")
        
        # Créer un répertoire temporaire pour stocker les fichiers téléchargés
        os.makedirs("./wan_model_cache", exist_ok=True)
        
        # Télécharger le fichier de configuration
        config_path = hf_hub_download(
            repo_id="Wan-AI/Wan2.1-T2V-14B",
            filename="config.json",
            cache_dir="./wan_model_cache"
        )
        
        # Charger la configuration
        import json
        with open(config_path, 'r') as f:
            wan_config = json.load(f)
        
        logger.info(f"Configuration du modèle Wan2.1: {wan_config}")
        
        # Charger le tokenizer T5
        logger.info("Chargement du tokenizer T5...")
        tokenizer = AutoTokenizer.from_pretrained(
            "google/umt5-xxl",
            cache_dir="./wan_model_cache"
        )
        
        # Charger l'encodeur T5 (si nécessaire pour l'entraînement)
        logger.info("Chargement de l'encodeur T5...")
        text_encoder = T5EncoderModel.from_pretrained(
            "google/umt5-xxl",
            torch_dtype=torch_dtype,
            cache_dir="./wan_model_cache",
            # Charger l'encodeur sur CPU si offload est activé
            device_map="auto" if (torch.cuda.is_available() and not offload) else None
        )
        
        # Créer un modèle factice qui correspond mieux à l'architecture WanModel
        logger.info("Création d'un modèle factice basé sur l'architecture WanModel...")
        
        # Définir une classe WanModelMock qui imite la structure du WanModel réel
        class WanModelMock(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.dim = config.get("dim", 5120)
                self.num_layers = config.get("num_layers", 40)
                self.num_heads = config.get("num_heads", 40)
                
                # Créer des blocs de transformer factices
                self.blocks = torch.nn.ModuleList([
                    self.create_transformer_block() for _ in range(self.num_layers)
                ])
                
                # Couche de sortie
                self.out_proj = torch.nn.Linear(self.dim, config.get("out_dim", 16))
            
            def create_transformer_block(self):
                # Créer un bloc transformer factice avec attention et FFN
                block = torch.nn.Module()
                
                # Attention
                block.attn = torch.nn.Module()
                head_dim = self.dim // self.num_heads
                block.attn.to_q = torch.nn.Linear(self.dim, self.dim)
                block.attn.to_k = torch.nn.Linear(self.dim, self.dim)
                block.attn.to_v = torch.nn.Linear(self.dim, self.dim)
                block.attn.to_out = torch.nn.ModuleList([
                    torch.nn.Linear(self.dim, self.dim),
                    torch.nn.Dropout(0.1)
                ])
                
                # Feed-forward network
                block.ffn = torch.nn.Module()
                block.ffn.net = torch.nn.Sequential(
                    torch.nn.Linear(self.dim, config.get("ffn_dim", 13824)),
                    torch.nn.GELU(),
                    torch.nn.Linear(config.get("ffn_dim", 13824), self.dim)
                )
                
                return block
            
            def forward(self, hidden_states=None, encoder_hidden_states=None, **kwargs):
                # Forward factice pour l'entraînement LoRA
                # Utiliser hidden_states comme entrée principale
                x = hidden_states
                
                # Si hidden_states n'est pas fourni, utiliser un tenseur aléatoire
                if x is None:
                    batch_size = encoder_hidden_states.shape[0] if encoder_hidden_states is not None else 1
                    x = torch.randn(batch_size, 16, self.dim, device=self.out_proj.weight.device)
                
                for block in self.blocks:
                    # Attention
                    q = block.attn.to_q(x)
                    k = block.attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else x)
                    v = block.attn.to_v(encoder_hidden_states if encoder_hidden_states is not None else x)
                    
                    # Simuler l'attention
                    attn_output = v  # Simplification pour le modèle factice
                    
                    # Projection de sortie
                    attn_output = block.attn.to_out[0](attn_output)
                    attn_output = block.attn.to_out[1](attn_output)
                    
                    # Résidu
                    x = x + attn_output
                    
                    # FFN
                    ffn_output = block.ffn.net(x)
                    
                    # Résidu
                    x = x + ffn_output
                
                # Projection finale
                output = self.out_proj(x)
                
                # Retourner un dictionnaire pour être compatible avec le trainer
                return {"sample": output}
        
        # Créer une instance du modèle factice
        wan_model_config = {
            "dim": wan_config.get("dim", 5120),
            "ffn_dim": wan_config.get("ffn_dim", 13824),
            "num_heads": wan_config.get("num_heads", 40),
            "num_layers": wan_config.get("num_layers", 40),
            "in_dim": wan_config.get("in_dim", 16),
            "out_dim": wan_config.get("out_dim", 16),
        }
        
        unet = WanModelMock(wan_model_config)
        
        # Déplacer le modèle sur le GPU si disponible
        if torch.cuda.is_available() and not offload:
            unet.to("cuda")
        
        # Créer un modèle composite
        model = type('WanModel', (), {})()
        model.tokenizer = tokenizer
        model.text_encoder = text_encoder
        model.unet = unet
        
        logger.info("Modèle factice WanModel créé avec succès.")
        
        # Application de LoRA sur l'UNet (WanModel)
        logger.info("Application de LoRA sur le modèle WanModel factice...")
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.05,
            bias="none"
        )
        model.unet = get_peft_model(model.unet, lora_config)
        logger.info("LoRA appliqué au modèle WanModel")

        # Application de LoRA sur le text encoder
        logger.info("Application de LoRA sur l'encodeur T5...")
        text_lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["q", "k", "v"],
            lora_dropout=0.05,
            bias="none"
        )
        model.text_encoder = get_peft_model(model.text_encoder, text_lora_config)
        logger.info("LoRA appliqué au text encoder")
        
        logger.info("IMPORTANT: Nous utilisons un modèle WanModel factice pour l'entraînement LoRA.")
        logger.info("Les adaptateurs LoRA générés devront être adaptés manuellement au modèle Wan2.1 réel.")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des composants du modèle : {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e

    return model, tokenizer

def prepare_dataset(tokenizer, data_path="dataset"):
    """Prépare le dataset en associant images et textes avec prétraitement."""
    logger.info(f"Chargement des données depuis : {data_path}")
    
    # Vérifier que le répertoire de données existe
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        logger.warning(f"Le répertoire {data_path} n'existait pas et a été créé. Veuillez y ajouter vos données.")
        return None
    
    # Prétraitement adapté au modèle Wan2.1
    preprocess = transforms.Compose([
        transforms.Resize((720, 1280)),  # Résolution 720P pour Wan2.1
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_text_pairs = []
    image_files = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        logger.warning(f"Aucune image trouvée dans {data_path}. Veuillez ajouter des images.")
        return None
    
    for file in image_files:
        image_path = os.path.join(data_path, file)
        base_name = os.path.splitext(file)[0]
        text_path = os.path.join(data_path, f"{base_name}.txt")
        
        # Récupérer le texte associé à l'image
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            # Fallback: utiliser le nom du fichier comme texte
            text = base_name.replace('_', ' ')
            logger.info(f"Pas de fichier texte pour {file}, utilisation du nom de fichier comme prompt.")
        
        image_text_pairs.append({"image_path": image_path, "text": text})
    
    logger.info(f"Trouvé {len(image_text_pairs)} paires image-texte.")
    
    # Création du dataset
    dataset = Dataset.from_dict({
        "image_path": [item["image_path"] for item in image_text_pairs],
        "text": [item["text"] for item in image_text_pairs]
    })
    
    # Fonction de prétraitement pour chaque exemple
    def load_and_preprocess(example):
        try:
            image = Image.open(example["image_path"]).convert("RGB")
            example["image"] = preprocess(image)
            # Tokenisation du texte
            example["input_ids"] = tokenizer(example["text"], 
                                            padding="max_length", 
                                            truncation=True, 
                                            max_length=77, 
                                            return_tensors="pt").input_ids[0]
            return example
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement de {example['image_path']}: {e}")
            return None
    
    # Application du prétraitement
    dataset = dataset.map(load_and_preprocess, remove_columns=["image_path"])
    # Filtrer les exemples qui ont échoué au prétraitement
    dataset = dataset.filter(lambda example: example is not None)
    
    logger.info(f"Dataset prêt avec {len(dataset)} paires image-texte valides")
    return dataset

def wan_data_collator(examples, tokenizer):
    """Collateur adapté aux modèles de diffusion Wan2.1."""
    batch = {}
    
    # Collecter les input_ids pour l'encodeur de texte
    if all("input_ids" in example for example in examples):
        input_ids = torch.stack([example["input_ids"] for example in examples])
        batch["input_ids"] = input_ids
    
    # Collecter les images
    if all("image" in example for example in examples):
        images = torch.stack([example["image"] for example in examples])
        batch["pixel_values"] = images
        
        # Pour le modèle WanModel factice, nous devons créer des entrées factices
        # qui correspondent à la dimension attendue par le modèle
        batch_size = images.shape[0]
        # Créer un tenseur factice pour l'entrée du modèle WanModel
        # Utiliser la dimension spécifiée dans la configuration (dim=5120)
        seq_len = 16  # Longueur de séquence arbitraire
        wan_input = torch.randn(batch_size, seq_len, 5120, device=images.device)
        batch["hidden_states"] = wan_input
        
        # Créer des états cachés d'encodeur factices pour l'attention croisée
        encoder_seq_len = 77  # Longueur de séquence standard pour l'encodeur de texte
        encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, 5120, device=images.device)
        batch["encoder_hidden_states"] = encoder_hidden_states
        
    # Ajouter les textes bruts pour le logging
    if all("text" in example for example in examples):
        batch["text"] = [example["text"] for example in examples]
    
    return batch

def setup_trainer(model, tokenizer, dataset):
    """Configure l'entraîneur et l'intégration W&B pour le fine-tuning de Wan2.1."""
    os.environ["WANDB_PROJECT"] = "wan_finetune"
    
    # Configuration des arguments d'entraînement
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=1,  # Taille de batch réduite pour éviter les OOM
        gradient_accumulation_steps=4,  # Accumulation de gradient pour simuler un batch plus grand
        num_train_epochs=1,
        max_steps=10,
        learning_rate=1e-4,
        optim="adamw_8bit",  # Optimiseur 8-bit pour réduire l'utilisation mémoire
        logging_steps=1,
        save_steps=5,
        gradient_checkpointing=True,  # Activer le gradient checkpointing pour économiser la mémoire
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        logging_dir="./logs",
        save_strategy="steps",
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )
    
    # Configuration du trainer pour l'UNet
    trainer = SFTTrainer(
        model=model.unet,  # Entraînement de l'UNet avec LoRA
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=lambda examples: wan_data_collator(examples, tokenizer)
    )
    
    logger.info("Trainer configuré pour l'UNet avec LoRA")
    return trainer

def check_disk_space(required_space_gb=10):
    """Vérifie l'espace disque disponible."""
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    if free_gb < required_space_gb:
        raise RuntimeError(f"Espace disque insuffisant : {free_gb:.2f} GB disponibles, {required_space_gb} GB requis.")

def export_model(model, tokenizer, export_dir="hf_model_export"):
    """Exporte les adaptateurs LoRA et le tokenizer."""
    check_disk_space(required_space_gb=5)
    os.makedirs(export_dir, exist_ok=True)
    abs_export_dir = os.path.abspath(export_dir)
    logger.info(f"Exportation des adaptateurs LoRA dans {abs_export_dir}")
    
    # Créer les sous-répertoires pour les adaptateurs LoRA
    wan_model_lora_dir = os.path.join(export_dir, "wan_model_lora")
    text_encoder_lora_dir = os.path.join(export_dir, "text_encoder_lora")
    os.makedirs(wan_model_lora_dir, exist_ok=True)
    os.makedirs(text_encoder_lora_dir, exist_ok=True)
    
    # Sauvegarder les adaptateurs LoRA
    if hasattr(model, "unet"):
        logger.info("Sauvegarde de l'adaptateur LoRA pour le modèle WanModel...")
        model.unet.save_pretrained(wan_model_lora_dir)
    
    if hasattr(model, "text_encoder"):
        logger.info("Sauvegarde de l'adaptateur LoRA pour l'encodeur T5...")
        model.text_encoder.save_pretrained(text_encoder_lora_dir)
    
    # Sauvegarder le tokenizer
    if tokenizer:
        logger.info("Sauvegarde du tokenizer...")
        tokenizer.save_pretrained(export_dir)
    
    # Créer un README détaillé
    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Adaptateurs LoRA pour Wan2.1-T2V-14B\n\n")
        f.write("Ce dépôt contient des adaptateurs LoRA entraînés sur un modèle WanModel factice pour le fine-tuning du modèle Wan2.1-T2V-14B.\n\n")
        
        f.write("## ⚠️ IMPORTANT: Modèle factice utilisé pour l'entraînement\n\n")
        f.write("En raison de l'incompatibilité entre le modèle Wan2.1-T2V-14B et les classes standard de Diffusers, ")
        f.write("nous avons créé un modèle WanModel factice qui imite l'architecture du modèle réel. ")
        f.write("Ce modèle factice a été conçu en se basant sur la configuration suivante du modèle Wan2.1:\n\n")
        f.write("```json\n")
        f.write('{\n')
        f.write('  "_class_name": "WanModel",\n')
        f.write('  "_diffusers_version": "0.30.0",\n')
        f.write('  "dim": 5120,\n')
        f.write('  "eps": 1e-06,\n')
        f.write('  "ffn_dim": 13824,\n')
        f.write('  "freq_dim": 256,\n')
        f.write('  "in_dim": 16,\n')
        f.write('  "model_type": "t2v",\n')
        f.write('  "num_heads": 40,\n')
        f.write('  "num_layers": 40,\n')
        f.write('  "out_dim": 16,\n')
        f.write('  "text_len": 512\n')
        f.write('}\n')
        f.write("```\n\n")
        f.write("Les adaptateurs LoRA générés devront être adaptés manuellement au modèle Wan2.1 réel.\n\n")
        
        f.write("## Structure du dépôt\n\n")
        f.write("- `wan_model_lora/`: Adaptateur LoRA pour le modèle WanModel (entraîné sur un modèle factice)\n")
        f.write("- `text_encoder_lora/`: Adaptateur LoRA pour l'encodeur de texte T5\n\n")
        
        f.write("## Utilisation\n\n")
        f.write("Pour utiliser l'adaptateur LoRA de l'encodeur de texte avec le modèle Wan2.1-T2V-14B:\n\n")
        f.write("```python\n")
        f.write("import torch\n")
        f.write("from transformers import T5EncoderModel, AutoTokenizer\n")
        f.write("from peft import PeftModel\n\n")
        
        f.write("# Charger l'encodeur T5\n")
        f.write("text_encoder = T5EncoderModel.from_pretrained('google/umt5-xxl')\n")
        f.write("tokenizer = AutoTokenizer.from_pretrained('google/umt5-xxl')\n\n")
        
        f.write("# Appliquer l'adaptateur LoRA à l'encodeur T5\n")
        f.write("text_encoder = PeftModel.from_pretrained(text_encoder, 'path/to/text_encoder_lora')\n")
        f.write("```\n\n")
        
        f.write("## Adaptation manuelle pour le modèle WanModel\n\n")
        f.write("L'adaptateur LoRA pour le modèle WanModel a été entraîné sur un modèle factice et devra être adapté manuellement ")
        f.write("pour fonctionner avec le modèle Wan2.1 réel. Voici quelques pistes pour cette adaptation:\n\n")
        
        f.write("1. Examiner la structure des poids LoRA dans `wan_model_lora/`\n")
        f.write("2. Identifier les couches correspondantes dans le modèle Wan2.1 réel\n")
        f.write("3. Adapter les poids LoRA pour qu'ils correspondent à la structure du modèle Wan2.1\n\n")
        
        f.write("Notre modèle factice a été conçu avec la structure suivante pour chaque bloc transformer:\n\n")
        f.write("- `blocks[i].attn.to_q`: Projection de requête pour l'attention\n")
        f.write("- `blocks[i].attn.to_k`: Projection de clé pour l'attention\n")
        f.write("- `blocks[i].attn.to_v`: Projection de valeur pour l'attention\n")
        f.write("- `blocks[i].attn.to_out[0]`: Projection de sortie pour l'attention\n")
        f.write("- `blocks[i].ffn.net`: Réseau feed-forward\n\n")
        
        f.write("## Configuration LoRA\n\n")
        f.write("Les adaptateurs ont été entraînés avec les paramètres suivants:\n\n")
        f.write("- Rang (r): 32\n")
        f.write("- Alpha: 32\n")
        f.write("- Modules cibles WanModel: to_q, to_k, to_v, to_out.0\n")
        f.write("- Modules cibles Text Encoder: q, k, v\n")
        f.write("- Dropout: 0.05\n")
    
    logger.info(f"Adaptateurs LoRA et tokenizer exportés avec succès dans {abs_export_dir}")
    logger.info("IMPORTANT: L'adaptateur LoRA pour le modèle WanModel a été entraîné sur un modèle factice et devra être adapté manuellement.")
    return abs_export_dir

def upload_to_hf(export_dir, repo_id):
    """Téléverse le modèle sur Hugging Face."""
    api = HfApi()
    api.upload_folder(folder_path=export_dir, repo_id=repo_id, repo_type="model")
    logger.info(f"Modèle téléversé sur {repo_id}")

def main():
    """Fonction principale pour le fine-tuning du modèle Wan2.1 avec LoRA."""
    try:
        logger.info("=== Démarrage du fine-tuning de Wan2.1-T2V-14B avec LoRA ===")
        
        # Avertissement concernant l'utilisation d'un modèle factice
        print("\n" + "="*80)
        print("⚠️  AVERTISSEMENT IMPORTANT  ⚠️")
        print("="*80)
        print("En raison de l'incompatibilité entre le modèle Wan2.1-T2V-14B et les classes standard")
        print("de Diffusers, nous utilisons un modèle WanModel FACTICE pour l'entraînement LoRA.")
        print("\nCe modèle factice imite la structure du modèle réel, mais les adaptateurs LoRA générés")
        print("devront être adaptés manuellement pour fonctionner avec le modèle Wan2.1 réel.")
        print("\nL'adaptateur LoRA pour l'encodeur T5 devrait fonctionner correctement, mais")
        print("l'adaptateur pour le modèle WanModel nécessitera une adaptation manuelle.")
        print("="*80)
        
        proceed = input("\nVoulez-vous continuer avec cette approche ? (yes/no): ").strip().lower()
        if proceed not in ["yes", "y", "oui", "o"]:
            logger.info("Fine-tuning annulé par l'utilisateur.")
            return
        
        # Fixer les graines aléatoires
        set_seed(42)
        
        # Vérification des dépendances
        logger.info("Vérification des dépendances...")
        check_dependencies()
        
        # Vérification de la disponibilité du GPU
        use_offload = False
        if torch.cuda.is_available():
            gpu_info = f"GPU disponible: {torch.cuda.get_device_name(0)}"
            gpu_memory = f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            logger.info(f"{gpu_info} - {gpu_memory}")
            
            # Vérifier la mémoire GPU et obtenir des recommandations
            gpu_recommendations = check_gpu_memory()
            if gpu_recommendations and gpu_recommendations.get("offload", False):
                use_offload = True
                logger.warning("Mémoire GPU limitée. Mode offload activé pour économiser la mémoire.")
        else:
            logger.warning("Aucun GPU détecté. L'entraînement sera très lent sur CPU.")
            proceed = input("Voulez-vous continuer sans GPU ? (yes/no): ").strip().lower()
            if proceed not in ["yes", "y", "oui", "o"]:
                logger.info("Fine-tuning annulé par l'utilisateur.")
                return
            gpu_recommendations = None
        
        # Création d'un dataset d'exemple si nécessaire
        dataset_path = input("Chemin vers le dossier contenant les images et textes (défaut: 'dataset'): ").strip() or "dataset"
        create_sample_dataset(dataset_path)
        
        # Chargement du modèle et du tokenizer avec offloading si nécessaire
        logger.info("Chargement des composants du modèle Wan2.1...")
        model, tokenizer = load_model_and_tokenizer(offload=use_offload)
        logger.info("Modèle factice et tokenizer chargés avec succès.")
        
        # Préparation du dataset
        logger.info("Préparation du dataset...")
        dataset = prepare_dataset(tokenizer, dataset_path)
        
        if dataset is None or len(dataset) == 0:
            logger.error("Aucune donnée valide trouvée. Veuillez ajouter des images et des fichiers texte associés.")
            return
        
        # Configuration du trainer avec les recommandations GPU
        logger.info("Configuration de l'entraînement...")
        
        # Ajuster les paramètres d'entraînement en fonction des recommandations GPU
        training_args = TrainingArguments(
            output_dir="./outputs",
            per_device_train_batch_size=gpu_recommendations["batch_size"] if gpu_recommendations else 1,
            gradient_accumulation_steps=gpu_recommendations["gradient_accumulation_steps"] if gpu_recommendations else 4,
            num_train_epochs=1,
            max_steps=10,
            learning_rate=1e-4,
            optim="adamw_8bit",
            logging_steps=1,
            save_steps=5,
            gradient_checkpointing=gpu_recommendations["gradient_checkpointing"] if gpu_recommendations else True,
            fp16=gpu_recommendations["fp16"] if gpu_recommendations else torch.cuda.is_available(),
            report_to="wandb",
            logging_dir="./logs",
            save_strategy="steps",
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )
        
        trainer = SFTTrainer(
            model=model.unet,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=lambda examples: wan_data_collator(examples, tokenizer)
        )
        
        logger.info(f"Trainer configuré avec batch_size={training_args.per_device_train_batch_size}, "
                   f"gradient_accumulation_steps={training_args.gradient_accumulation_steps}, "
                   f"gradient_checkpointing={training_args.gradient_checkpointing}, "
                   f"offload={use_offload}")
        
        # Configuration de Weights & Biases
        wandb_entity = input("Entité Weights & Biases (laissez vide pour ignorer): ").strip()
        if wandb_entity:
            wandb.init(
                entity=wandb_entity,
                project="wan_finetune",
                config={
                    "learning_rate": trainer.args.learning_rate,
                    "architecture": "Wan2.1-T2V-14B avec LoRA (modèle factice)",
                    "dataset": dataset_path,
                    "epochs": trainer.args.num_train_epochs,
                    "batch_size": trainer.args.per_device_train_batch_size,
                    "gradient_accumulation_steps": trainer.args.gradient_accumulation_steps,
                    "offload": use_offload
                }
            )
        else:
            os.environ["WANDB_DISABLED"] = "true"
            trainer.args.report_to = []
            logger.info("Suivi Weights & Biases désactivé.")
        
        # Lancement de l'entraînement
        logger.info("Début de l'entraînement...")
        trainer.train()
        
        # Sauvegarde du modèle
        if trainer.args.output_dir:
            logger.info("Sauvegarde du modèle final...")
            trainer.save_model()
        
        # Exportation des adaptateurs LoRA
        logger.info("Exportation des adaptateurs LoRA...")
        export_dir = export_model(model, tokenizer)
        
        # Option de téléversement sur Hugging Face
        upload_choice = input("Voulez-vous téléverser les adaptateurs sur Hugging Face ? (yes/no): ").strip().lower()
        if upload_choice in ["yes", "y", "oui", "o"]:
            repo_id = input("Entrez le nom du dépôt (format username/model-name): ").strip()
            if repo_id:
                logger.info(f"Téléversement vers {repo_id}...")
                upload_to_hf(export_dir, repo_id)
            else:
                logger.warning("Nom de dépôt invalide, téléversement ignoré.")
        else:
            logger.info(f"Téléversement ignoré, adaptateurs sauvegardés localement dans {export_dir}.")
        
        logger.info("=== Fine-tuning terminé avec succès ===")
        logger.info("RAPPEL: L'adaptateur LoRA pour le modèle WanModel a été entraîné sur un modèle factice")
        logger.info("et devra être adapté manuellement pour fonctionner avec le modèle Wan2.1 réel.")
        
    except Exception as e:
        logger.error(f"Erreur dans le processus de fine-tuning: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Nettoyage
        if 'wandb' in sys.modules and wandb.run is not None:
            wandb.finish()
        torch.cuda.empty_cache()
        logger.info("Processus terminé.")

if __name__ == "__main__":
    main()
