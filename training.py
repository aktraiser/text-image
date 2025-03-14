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

from transformers import CLIPTokenizer, CLIPTextModel, TrainingArguments
from peft import LoraConfig, get_peft_model
from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline
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
    required = ["torch", "transformers", "peft", "diffusers", "trl", "datasets", "wandb", "numpy", "pillow", "bitsandbytes"]
    for pkg in required:
        if importlib.util.find_spec(pkg) is None:
            logger.warning(f"{pkg} n'est pas installé. Installation en cours...")
            subprocess.run(["pip", "install", pkg], check=True)
            logger.info(f"{pkg} installé avec succès.")

def load_model_and_tokenizer(offload=False):
    """Charge les composants de Stable Diffusion et applique LoRA.
    
    Args:
        offload (bool): Si True, certains composants seront chargés sur CPU pour économiser la mémoire GPU.
    """
    check_dependencies()
    
    # Déterminer le type de données à utiliser
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() and not offload else "cpu"
    
    try:
        logger.info("Chargement des composants de Stable Diffusion...")
        logger.info(f"Mode d'offloading: {'activé' if offload else 'désactivé'}")
        logger.info(f"Type de données initial: {torch_dtype}")
        
        # Utiliser Stable Diffusion 2.1
        model_id = "stabilityai/stable-diffusion-2-1"
        
        # Chargement du pipeline complet pour extraire les composants
        logger.info("Chargement du pipeline Stable Diffusion...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant="fp16" if torch.cuda.is_available() else None,
        )
        
        # Extraire les composants individuels
        logger.info("Extraction des composants individuels...")
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet = pipe.unet
        
        # Libérer la mémoire du pipeline complet
        del pipe
        torch.cuda.empty_cache()
        
        # Créer un modèle composite pour l'entraînement LoRA
        model = type('DiffusionModel', (), {})()
        model.unet = unet
        model.text_encoder = text_encoder
        model.tokenizer = tokenizer
        model.vae = vae
        
        logger.info("Composants de Stable Diffusion extraits avec succès.")
        
        # Application de LoRA sur l'UNet
        logger.info("Application de LoRA sur l'UNet...")
        lora_config = LoraConfig(
            r=64,  # Augmenté de 32 à 64 pour une meilleure capacité d'adaptation
            lora_alpha=64,  # Augmenté de 32 à 64 pour correspondre au rang
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "conv1", "conv2"],  # Ajout de modules supplémentaires pour une meilleure adaptation
            lora_dropout=0.05,
            bias="none"
        )
        model.unet = get_peft_model(model.unet, lora_config)
        logger.info("LoRA appliqué à l'UNet")

        # Application de LoRA sur le text encoder
        logger.info("Application de LoRA sur l'encodeur de texte...")
        text_lora_config = LoraConfig(
            r=64,  # Augmenté de 32 à 64 pour une meilleure capacité d'adaptation
            lora_alpha=64,  # Augmenté de 32 à 64 pour correspondre au rang
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],  # Ajout de modules supplémentaires pour une meilleure adaptation
            lora_dropout=0.05,
            bias="none"
        )
        model.text_encoder = get_peft_model(model.text_encoder, text_lora_config)
        logger.info("LoRA appliqué à l'encodeur de texte")
        
        # Déplacer les modèles sur le GPU si disponible et si l'offloading n'est pas activé
        if torch.cuda.is_available() and not offload:
            logger.info("Déplacement des modèles sur GPU...")
            model.unet = model.unet.to("cuda")
            model.text_encoder = model.text_encoder.to("cuda")
            model.vae = model.vae.to("cuda")
            
            # S'assurer que tous les modèles sont du même type de données
            logger.info(f"Conversion des modèles en {torch_dtype}...")
            model.unet = model.unet.to(dtype=torch_dtype)
            model.text_encoder = model.text_encoder.to(dtype=torch_dtype)
            model.vae = model.vae.to(dtype=torch_dtype)
            
            # Vérifier les types de données après conversion
            logger.info(f"Types de données après conversion - UNet: {model.unet.dtype if hasattr(model.unet, 'dtype') else 'N/A'}, "
                       f"Text Encoder: {model.text_encoder.dtype if hasattr(model.text_encoder, 'dtype') else 'N/A'}, "
                       f"VAE: {model.vae.dtype if hasattr(model.vae, 'dtype') else 'N/A'}")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
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
    
    # Prétraitement standard pour les modèles de diffusion
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
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
            
            # Tokenisation du texte avec attention_mask
            tokenized = tokenizer(
                example["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=77, 
                return_tensors="pt"
            )
            
            # Extraire les tenseurs et les convertir en tenseurs 1D (enlever la dimension batch)
            example["input_ids"] = tokenized.input_ids[0]
            example["attention_mask"] = tokenized.attention_mask[0]
            
            # S'assurer que les tenseurs sont sur CPU, détachés et en float32
            # Nous convertirons en float16 si nécessaire lors de l'entraînement
            example["input_ids"] = example["input_ids"].cpu().detach()
            example["attention_mask"] = example["attention_mask"].cpu().detach()
            example["image"] = example["image"].cpu().detach().float()  # Assurer que l'image est en float32
            
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

def data_collator(examples, tokenizer):
    """Collateur adapté aux modèles de diffusion standard."""
    batch = {}
    
    # Collecter les input_ids pour l'encodeur de texte
    if all("input_ids" in example for example in examples):
        # Vérifier si input_ids est déjà un tenseur ou une liste
        if isinstance(examples[0]["input_ids"], torch.Tensor):
            input_ids = torch.stack([example["input_ids"] for example in examples])
        else:
            # Convertir les listes en tenseurs avant de les empiler
            input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in examples])
        batch["input_ids"] = input_ids
    
    # Collecter les attention_mask
    if all("attention_mask" in example for example in examples):
        # Vérifier si attention_mask est déjà un tenseur ou une liste
        if isinstance(examples[0]["attention_mask"], torch.Tensor):
            attention_mask = torch.stack([example["attention_mask"] for example in examples])
        else:
            # Convertir les listes en tenseurs avant de les empiler
            attention_mask = torch.stack([torch.tensor(example["attention_mask"]) for example in examples])
        batch["attention_mask"] = attention_mask
    
    # Collecter les images
    if all("image" in example for example in examples):
        # Vérifier si image est déjà un tenseur
        if isinstance(examples[0]["image"], torch.Tensor):
            images = torch.stack([example["image"] for example in examples])
        else:
            # Convertir en tenseur si nécessaire
            images = torch.stack([torch.tensor(example["image"]) for example in examples])
        batch["pixel_values"] = images
        
    # Ajouter les textes bruts pour le logging
    if all("text" in example for example in examples):
        batch["text"] = [example["text"] for example in examples]
    
    return batch

def setup_trainer(model, tokenizer, dataset):
    """Configure l'entraîneur et l'intégration W&B pour le fine-tuning."""
    os.environ["WANDB_PROJECT"] = "diffusion_finetune"
    
    # Configuration des arguments d'entraînement
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=1,  # Taille de batch réduite pour éviter les OOM
        gradient_accumulation_steps=4,  # Accumulation de gradient pour simuler un batch plus grand
        num_train_epochs=100,  # Augmenté de 1 à 100 pour un meilleur apprentissage
        max_steps=500,  # Augmenté de 10 à 500 pour un apprentissage plus approfondi
        learning_rate=1e-4,
        optim="adamw_8bit",  # Optimiseur 8-bit pour réduire l'utilisation mémoire
        logging_steps=10,  # Augmenté pour réduire la verbosité des logs
        save_steps=50,  # Augmenté pour sauvegarder moins fréquemment
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
    
    return training_args

def train_diffusion_model(model, tokenizer, dataset, training_args):
    """Fonction d'entraînement personnalisée pour les modèles de diffusion avec LoRA."""
    logger.info("Configuration de l'entraînement personnalisée pour le modèle de diffusion...")
    
    try:
        # Création du DataLoader
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: data_collator(examples, tokenizer),
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
        
        # Vérifier que le DataLoader fonctionne en essayant de charger un batch
        logger.info("Vérification du DataLoader...")
        test_batch = next(iter(train_dataloader))
        logger.info(f"Test de chargement de données réussi. Forme des données: {[k + ': ' + str(v.shape) if isinstance(v, torch.Tensor) else k + ': ' + str(type(v)) for k, v in test_batch.items()]}")
        
        # Configuration de l'optimiseur
        from transformers import get_scheduler
        import bitsandbytes as bnb
        
        # Paramètres à optimiser (uniquement les paramètres LoRA)
        unet_params = [p for n, p in model.unet.named_parameters() if "lora" in n and p.requires_grad]
        text_encoder_params = [p for n, p in model.text_encoder.named_parameters() if "lora" in n and p.requires_grad]
        
        logger.info(f"Nombre de paramètres LoRA à optimiser - UNet: {len(unet_params)}, Text Encoder: {len(text_encoder_params)}")
        
        # Déterminer le type de données à utiliser
        weight_dtype = torch.float16 if training_args.fp16 else torch.float32
        
        # Désactiver fp16 si on utilise l'optimiseur 8-bit pour éviter les conflits
        if "adamw_8bit" in training_args.optim.lower():
            logger.info("Optimiseur 8-bit détecté, désactivation de fp16 pour éviter les conflits")
            training_args.fp16 = False
            weight_dtype = torch.float32
            
            # Convertir tous les modèles en float32 pour assurer la compatibilité avec l'optimiseur 8-bit
            logger.info("Conversion de tous les modèles en float32 pour compatibilité avec l'optimiseur 8-bit")
            model.unet = model.unet.to(dtype=torch.float32)
            model.text_encoder = model.text_encoder.to(dtype=torch.float32)
            model.vae = model.vae.to(dtype=torch.float32)
        
        # Utiliser l'optimiseur 8-bit AdamW pour économiser la mémoire
        if "adamw_8bit" in training_args.optim.lower():
            logger.info("Utilisation de l'optimiseur AdamW 8-bit")
            optimizer = bnb.optim.AdamW8bit(
                unet_params + text_encoder_params,
                lr=training_args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
        else:
            logger.info("Utilisation de l'optimiseur AdamW standard")
            optimizer = torch.optim.AdamW(
                unet_params + text_encoder_params,
                lr=training_args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
        
        # Nombre total d'étapes d'entraînement
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            num_training_steps = len(train_dataloader) * training_args.num_train_epochs // training_args.gradient_accumulation_steps
        
        # Planificateur de taux d'apprentissage
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps,
        )
        
        # Préparation pour l'entraînement
        if training_args.gradient_checkpointing:
            model.unet.gradient_checkpointing_enable()
            model.text_encoder.gradient_checkpointing_enable()
        
        # Déplacer les modèles sur GPU si disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.unet.to(device)
        model.text_encoder.to(device)
        model.vae.to(device)
        
        # S'assurer que tous les modèles sont du même type de données
        logger.info(f"Vérification que tous les modèles sont en {weight_dtype}")
        model.unet = model.unet.to(dtype=weight_dtype)
        model.text_encoder = model.text_encoder.to(dtype=weight_dtype)
        model.vae = model.vae.to(dtype=weight_dtype)
        logger.info(f"Tous les modèles convertis en {weight_dtype}")
        
        # Mettre le VAE en mode évaluation car nous ne l'entraînons pas
        model.vae.eval()
        
        # Scaler pour l'entraînement en précision mixte
        scaler = torch.cuda.amp.GradScaler() if training_args.fp16 else None
        
        # Boucle d'entraînement
        logger.info("Début de l'entraînement...")
        global_step = 0
        
        # Créer les répertoires de sortie
        os.makedirs(training_args.output_dir, exist_ok=True)
        os.makedirs(training_args.logging_dir, exist_ok=True)
        
        # Fonction de bruit pour l'entraînement du modèle de diffusion
        from diffusers import DDPMScheduler
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        
        # Boucle principale d'entraînement
        for epoch in range(int(training_args.num_train_epochs)):
            model.unet.train()
            model.text_encoder.train()
            
            for step, batch in enumerate(train_dataloader):
                try:
                    # Vérifier si on a atteint le nombre maximum d'étapes
                    if global_step >= training_args.max_steps:
                        break
                    
                    # Déplacer les données sur le périphérique et convertir au bon type
                    pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    
                    # Convertir les images en latents
                    with torch.no_grad():
                        # Utiliser le même type de données pour le VAE et les entrées
                        latents = model.vae.encode(pixel_values).latent_dist.sample() * 0.18215
                    
                    # Ajouter du bruit aux latents
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Obtenir les embeddings de texte
                    with torch.no_grad():
                        encoder_hidden_states = model.text_encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )[0]
                    
                    # Vérifier et convertir les types de données si nécessaire
                    if encoder_hidden_states.dtype != weight_dtype:
                        logger.info(f"Conversion des encoder_hidden_states de {encoder_hidden_states.dtype} à {weight_dtype}")
                        encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)
                    
                    if noisy_latents.dtype != weight_dtype:
                        logger.info(f"Conversion des noisy_latents de {noisy_latents.dtype} à {weight_dtype}")
                        noisy_latents = noisy_latents.to(dtype=weight_dtype)
                    
                    if timesteps.dtype != torch.int64:
                        timesteps = timesteps.to(dtype=torch.int64)
                    
                    # Prédire le bruit
                    if training_args.fp16:
                        with torch.cuda.amp.autocast():
                            noise_pred = model.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                            loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    else:
                        noise_pred = model.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    
                    # Diviser la perte par le nombre d'étapes d'accumulation de gradient
                    loss = loss / training_args.gradient_accumulation_steps
                    
                    # Rétropropagation avec précision mixte si activée
                    if training_args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Mise à jour des poids après accumulation de gradient
                    if (step + 1) % training_args.gradient_accumulation_steps == 0:
                        if training_args.fp16:
                            # Avec précision mixte
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(unet_params + text_encoder_params, 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # Sans précision mixte
                            torch.nn.utils.clip_grad_norm_(unet_params + text_encoder_params, 1.0)
                            optimizer.step()
                        
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                        # Logging
                        if global_step % training_args.logging_steps == 0:
                            logger.info(f"Étape {global_step}/{num_training_steps} - Perte: {loss.item() * training_args.gradient_accumulation_steps:.4f}")
                            if "wandb" in training_args.report_to:
                                wandb.log({
                                    "loss": loss.item() * training_args.gradient_accumulation_steps,
                                    "lr": lr_scheduler.get_last_lr()[0],
                                    "step": global_step,
                                })
                        
                        # Sauvegarde du modèle
                        if global_step % training_args.save_steps == 0:
                            save_path = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                            os.makedirs(save_path, exist_ok=True)
                            
                            # Sauvegarder les adaptateurs LoRA
                            model.unet.save_pretrained(os.path.join(save_path, "unet_lora"))
                            model.text_encoder.save_pretrained(os.path.join(save_path, "text_encoder_lora"))
                            logger.info(f"Modèle sauvegardé à l'étape {global_step} dans {save_path}")
                
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du batch {step}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
        
        # Sauvegarde finale
        final_save_path = os.path.join(training_args.output_dir, "final")
        os.makedirs(final_save_path, exist_ok=True)
        model.unet.save_pretrained(os.path.join(final_save_path, "unet_lora"))
        model.text_encoder.save_pretrained(os.path.join(final_save_path, "text_encoder_lora"))
        logger.info(f"Entraînement terminé. Modèle final sauvegardé dans {final_save_path}")
        
        return model
    
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e

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
    unet_lora_dir = os.path.join(export_dir, "unet_lora")
    text_encoder_lora_dir = os.path.join(export_dir, "text_encoder_lora")
    os.makedirs(unet_lora_dir, exist_ok=True)
    os.makedirs(text_encoder_lora_dir, exist_ok=True)
    
    # Sauvegarder les adaptateurs LoRA
    if hasattr(model, "unet"):
        logger.info("Sauvegarde de l'adaptateur LoRA pour l'UNet...")
        model.unet.save_pretrained(unet_lora_dir)
    
    if hasattr(model, "text_encoder"):
        logger.info("Sauvegarde de l'adaptateur LoRA pour l'encodeur de texte...")
        model.text_encoder.save_pretrained(text_encoder_lora_dir)
    
    # Sauvegarder le tokenizer
    if tokenizer:
        logger.info("Sauvegarde du tokenizer...")
        tokenizer.save_pretrained(export_dir)
    
    # Créer un README détaillé
    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Adaptateurs LoRA pour Stable Diffusion\n\n")
        f.write("Ce dépôt contient des adaptateurs LoRA pour fine-tuner Stable Diffusion 2.1.\n\n")
        
        f.write("## Structure du dépôt\n\n")
        f.write("- `unet_lora/`: Adaptateur LoRA pour l'UNet\n")
        f.write("- `text_encoder_lora/`: Adaptateur LoRA pour l'encodeur de texte\n\n")
        
        f.write("## Utilisation\n\n")
        f.write("Pour utiliser ces adaptateurs avec Stable Diffusion:\n\n")
        f.write("```python\n")
        f.write("import torch\n")
        f.write("from diffusers import StableDiffusionPipeline\n")
        f.write("from peft import PeftModel\n\n")
        
        f.write("# Charger le modèle de base\n")
        f.write("model_id = 'stabilityai/stable-diffusion-2-1'\n")
        f.write("pipe = StableDiffusionPipeline.from_pretrained(model_id)\n\n")
        
        f.write("# Appliquer les adaptateurs LoRA\n")
        f.write("pipe.unet = PeftModel.from_pretrained(pipe.unet, 'path/to/unet_lora')\n")
        f.write("pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, 'path/to/text_encoder_lora')\n\n")
        
        f.write("# Déplacer sur GPU si disponible\n")
        f.write("pipe = pipe.to('cuda')\n\n")
        
        f.write("# Générer une image\n")
        f.write("prompt = 'Description de l\\'image souhaitée'\n")
        f.write("image = pipe(prompt=prompt).images[0]\n")
        f.write("image.save('image_resultat.png')\n")
        f.write("```\n\n")
        
        f.write("## Configuration LoRA\n\n")
        f.write("Les adaptateurs ont été entraînés avec les paramètres suivants:\n\n")
        f.write("- Rang (r): 64\n")
        f.write("- Alpha: 64\n")
        f.write("- Modules cibles UNet: to_q, to_k, to_v, to_out.0, proj_in, proj_out, conv1, conv2\n")
        f.write("- Modules cibles Text Encoder: q_proj, k_proj, v_proj, out_proj, fc1, fc2\n")
        f.write("- Dropout: 0.05\n")
    
    logger.info(f"Adaptateurs LoRA et tokenizer exportés avec succès dans {abs_export_dir}")
    return abs_export_dir

def upload_to_hf(export_dir, repo_id):
    """Téléverse le modèle sur Hugging Face."""
    api = HfApi()
    api.upload_folder(folder_path=export_dir, repo_id=repo_id, repo_type="model")
    logger.info(f"Modèle téléversé sur {repo_id}")

def main():
    """Fonction principale pour le fine-tuning avec LoRA."""
    try:
        logger.info("=== Démarrage du fine-tuning de Stable Diffusion avec LoRA ===")
        
        # Fixer les graines aléatoires
        set_seed(42)
        
        # Vérification des dépendances
        logger.info("Vérification des dépendances...")
        check_dependencies()
        
        # Option de débogage
        debug_mode = input("Activer le mode débogage pour plus d'informations ? (yes/no, défaut: no): ").strip().lower()
        if debug_mode in ["yes", "y", "oui", "o"]:
            logger.setLevel(logging.DEBUG)
            logger.debug("Mode débogage activé - Affichage des informations détaillées")
        
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
        logger.info("Chargement des composants du modèle...")
        model, tokenizer = load_model_and_tokenizer(offload=use_offload)
        logger.info("Modèle et tokenizer chargés avec succès.")
        
        # Préparation du dataset
        logger.info("Préparation du dataset...")
        dataset = prepare_dataset(tokenizer, dataset_path)
        
        if dataset is None or len(dataset) == 0:
            logger.error("Aucune donnée valide trouvée. Veuillez ajouter des images et des fichiers texte associés.")
            return
        
        # Configuration de l'entraînement
        logger.info("Configuration de l'entraînement...")
        
        # Choix de l'optimiseur
        optim_choice = "adamw"  # Par défaut, utiliser AdamW standard
        
        # Demander à l'utilisateur s'il souhaite utiliser l'optimiseur 8-bit
        use_8bit = input("Utiliser l'optimiseur 8-bit pour économiser la mémoire ? (yes/no, défaut: yes): ").strip().lower()
        if not use_8bit or use_8bit in ["yes", "y", "oui", "o"]:
            optim_choice = "adamw_8bit"
            logger.info("Optimiseur 8-bit sélectionné.")
            # Désactiver fp16 si on utilise l'optimiseur 8-bit pour éviter les conflits
            if gpu_recommendations:
                gpu_recommendations["fp16"] = False
                logger.info("Précision mixte (fp16) désactivée pour éviter les conflits avec l'optimiseur 8-bit.")
        else:
            logger.info("Optimiseur standard sélectionné.")
        
        # Forcer le type de données
        force_dtype = input("Forcer un type de données spécifique ? (float32/float16/auto, défaut: auto): ").strip().lower()
        if force_dtype == "float32":
            logger.info("Forçage du type de données à float32")
            # Convertir tous les modèles en float32
            model.unet = model.unet.to(dtype=torch.float32)
            model.text_encoder = model.text_encoder.to(dtype=torch.float32)
            model.vae = model.vae.to(dtype=torch.float32)
            if gpu_recommendations:
                gpu_recommendations["fp16"] = False
        elif force_dtype == "float16" and torch.cuda.is_available():
            logger.info("Forçage du type de données à float16")
            # Convertir tous les modèles en float16
            model.unet = model.unet.to(dtype=torch.float16)
            model.text_encoder = model.text_encoder.to(dtype=torch.float16)
            model.vae = model.vae.to(dtype=torch.float16)
            if gpu_recommendations:
                gpu_recommendations["fp16"] = True
        else:
            logger.info("Type de données automatique basé sur les recommandations")
        
        # Ajuster les paramètres d'entraînement en fonction des recommandations GPU
        training_args = TrainingArguments(
            output_dir="./outputs",
            per_device_train_batch_size=gpu_recommendations["batch_size"] if gpu_recommendations else 1,
            gradient_accumulation_steps=gpu_recommendations["gradient_accumulation_steps"] if gpu_recommendations else 4,
            num_train_epochs=1,
            max_steps=500,  # Changé de 10 à 500 pour un entraînement plus long
            learning_rate=1e-4,
            optim=optim_choice,
            logging_steps=10,  # Changé de 1 à 10 pour réduire la verbosité des logs
            save_steps=50,  # Augmenté de 5 à 50 pour sauvegarder moins fréquemment
            gradient_checkpointing=gpu_recommendations["gradient_checkpointing"] if gpu_recommendations else True,
            fp16=gpu_recommendations["fp16"] if gpu_recommendations else False,  # Désactivé par défaut pour éviter les conflits
            report_to="wandb",
            logging_dir="./logs",
            save_strategy="steps",
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )
        
        logger.info(f"Configuration d'entraînement: batch_size={training_args.per_device_train_batch_size}, "
                   f"gradient_accumulation_steps={training_args.gradient_accumulation_steps}, "
                   f"fp16={training_args.fp16}, optim={training_args.optim}")
        
        # Afficher les types de données des modèles avant l'entraînement
        logger.info("Types de données des modèles avant l'entraînement:")
        logger.info(f"UNet: {next(model.unet.parameters()).dtype}")
        logger.info(f"Text Encoder: {next(model.text_encoder.parameters()).dtype}")
        logger.info(f"VAE: {next(model.vae.parameters()).dtype}")
        
        # Configuration de Weights & Biases
        wandb_entity = input("Entité Weights & Biases (laissez vide pour ignorer): ").strip()
        if wandb_entity:
            wandb.init(
                entity=wandb_entity,
                project="diffusion_finetune",
                config={
                    "learning_rate": training_args.learning_rate,
                    "architecture": "Stable Diffusion 2.1 avec LoRA",
                    "dataset": dataset_path,
                    "epochs": training_args.num_train_epochs,
                    "batch_size": training_args.per_device_train_batch_size,
                    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                    "offload": use_offload,
                    "fp16": training_args.fp16,
                    "optim": training_args.optim
                }
            )
        else:
            os.environ["WANDB_DISABLED"] = "true"
            training_args.report_to = []
            logger.info("Suivi Weights & Biases désactivé.")
        
        # Lancement de l'entraînement personnalisé
        logger.info("Début de l'entraînement...")
        model = train_diffusion_model(model, tokenizer, dataset, training_args)
        
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
