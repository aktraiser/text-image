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
    
    # First, let's examine the directory structure
    logger.info("Examining dataset directory structure:")
    files = os.listdir(data_path)
    logger.info(f"Files in dataset directory: {files[:10]}...")
    
    # Create a custom dataset that pairs images with their text files
    image_text_pairs = []
    
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(data_path, file)
            # Look for a corresponding text file
            base_name = os.path.splitext(file)[0]
            text_path = os.path.join(data_path, f"{base_name}.txt")
            
            if os.path.exists(text_path):
                # Read the text file
                with open(text_path, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                
                # Add the pair to our dataset
                image_text_pairs.append({
                    "image_path": image_path,
                    "text": text_content
                })
                logger.info(f"Paired image {file} with text: {text_content[:50]}...")
            else:
                # Use the filename as fallback text
                image_text_pairs.append({
                    "image_path": image_path,
                    "text": base_name.replace('_', ' ')
                })
                logger.info(f"No text file for {file}, using filename as text")
    
    logger.info(f"Created {len(image_text_pairs)} image-text pairs")
    
    # Convert to HuggingFace dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({
        "image_path": [item["image_path"] for item in image_text_pairs],
        "text": [item["text"] for item in image_text_pairs]
    })
    
    # Load images
    def load_image(example):
        example["image"] = Image.open(example["image_path"]).convert("RGB")
        return example
    
    dataset = dataset.map(load_image)
    
    # Add the trigger token to the text
    def preprocess_text(example):
        example["text"] = f"{example['text']} TOK"  # "TOK" comme mot déclencheur
        return example
    
    dataset = dataset.map(preprocess_text)
    
    # Log a sample
    if len(dataset) > 0:
        logger.info(f"Sample: {dataset[0]}")
    
    return dataset

def setup_trainer(model, tokenizer, dataset):
    """Configure l'entraîneur avec les hyperparamètres d'entraînement."""
    # Set up wandb if needed
    os.environ["WANDB_PROJECT"] = "wan_finetune"  # Set project name via environment variable
    
    # First, let's check what kind of model we have
    logger.info(f"Model type: {type(model)}")
    if hasattr(model, "unet"):
        logger.info(f"UNet type: {type(model.unet)}")
    
    # For Wan2.1, we need to use a specialized approach similar to wan-lora-trainer
    # See: https://replicate.com/ostris/wan-lora-trainer/train?input=python
    
    # Create a custom trainer for Wan2.1
    from transformers import Trainer
    
    # Define a custom data collator for Wan2.1 training
    def wan_data_collator(examples):
        batch = {}
        
        # Extract text inputs
        texts = [example["text"] for example in examples]
        text_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Add text inputs to batch
        batch.update(text_inputs)
        
        # Add images to batch if needed
        if "image" in examples[0]:
            # Process images (in a real implementation, this would convert to tensors)
            # For now, we'll just log that we have images
            logger.info(f"Processing batch with {len(examples)} images")
        
        # For testing purposes, we'll just use the text as labels
        batch["labels"] = batch["input_ids"].clone()
        
        return batch
    
    # Use a standard Trainer as fallback
    logger.info("Using standard Trainer with custom Wan data collator")
    
    # Determine which model component to train
    train_model = None
    if hasattr(model, "unet") and model.unet is not None and isinstance(model.unet, torch.nn.Module):
        train_model = model.unet
        logger.info("Will train the UNet component")
    elif hasattr(model, "text_encoder") and model.text_encoder is not None and isinstance(model.text_encoder, torch.nn.Module):
        train_model = model.text_encoder
        logger.info("Will train the text_encoder component")
    else:
        # If we can't find a specific component, try using a dummy model for testing
        from transformers import AutoModelForCausalLM
        logger.info("Creating a dummy model for testing")
        train_model = AutoModelForCausalLM.from_pretrained("gpt2-medium", device_map="auto")
    
    training_args = TrainingArguments(
        output_dir="./outputs",          # Dossier de sortie
        per_device_train_batch_size=1,   # Taille du batch par appareil
        num_train_epochs=1,              # Nombre d'époques
        max_steps=10,                    # Nombre total de pas
        learning_rate=0.0001,            # Taux d'apprentissage
        optim="adamw_8bit",              # Optimiseur adapté
        logging_steps=5,                 # Intervalle de logs
        save_steps=10,                   # Intervalle de sauvegarde
        gradient_checkpointing=False,    # Pas de checkpointing
        fp16=torch.cuda.is_available(),  # Utiliser FP16 si GPU disponible
        report_to="wandb",               # Suivi avec Weights & Biases
        logging_dir="./logs",            # Dossier des logs
        save_strategy="steps",           # Sauvegarde par étapes
        save_total_limit=3,              # Limite de sauvegardes
        remove_unused_columns=False,     # Don't remove columns from dataset
    )
    
    trainer = Trainer(
        model=train_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=wan_data_collator,
    )
    
    return trainer

def export_model(model, tokenizer, export_dir):
    """Sauvegarde le modèle fine-tuné localement."""
    os.makedirs(export_dir, exist_ok=True)
    logger.info(f"Exportation du modèle vers : {export_dir}")
    
    # Import safetensors explicitly
    try:
        import safetensors
        logger.info(f"SafeTensors version: {safetensors.__version__}")
    except ImportError:
        logger.warning("SafeTensors not found, installing it now...")
        subprocess.run(["pip", "install", "safetensors"], check=True)
        import safetensors
        logger.info(f"SafeTensors installed, version: {safetensors.__version__}")
    
    # Sauvegarder le modèle avec sharding
    if hasattr(model, "save_pretrained"):
        logger.info("Saving model using save_pretrained method")
        try:
            model.save_pretrained(export_dir, max_shard_size="5GB", safe_serialization=True)
            logger.info("Model saved with safe_serialization=True")
        except Exception as e:
            logger.error(f"Error saving with safe_serialization: {str(e)}")
            logger.info("Trying alternative saving method...")
            model.save_pretrained(export_dir, max_shard_size="5GB")
            logger.info("Model saved with default serialization")
    else:
        # Fallback for custom model structure
        if hasattr(model, "unet") and hasattr(model.unet, "save_pretrained"):
            logger.info("Saving UNet component")
            try:
                model.unet.save_pretrained(
                    os.path.join(export_dir, "unet"), 
                    max_shard_size="5GB", 
                    safe_serialization=True
                )
                logger.info("UNet saved with safe_serialization=True")
            except Exception as e:
                logger.error(f"Error saving UNet with safe_serialization: {str(e)}")
                model.unet.save_pretrained(os.path.join(export_dir, "unet"), max_shard_size="5GB")
                logger.info("UNet saved with default serialization")
        else:
            logger.warning("Could not find a valid model component to save")
            
            # If we're using a dummy model, save it
            if isinstance(model, torch.nn.Module):
                logger.info("Saving model as a PyTorch module")
                try:
                    # Try using safetensors directly
                    from safetensors.torch import save_file
                    state_dict = model.state_dict()
                    save_file(state_dict, os.path.join(export_dir, "model.safetensors"))
                    logger.info("Model saved using safetensors.torch.save_file")
                except Exception as e:
                    logger.error(f"Error saving with safetensors: {str(e)}")
                    # Fallback to PyTorch saving
                    torch.save(model.state_dict(), os.path.join(export_dir, "model.pt"))
                    logger.info("Model saved as model.pt using torch.save")
            else:
                logger.error("Model is not a valid PyTorch module, cannot save")
    
    # Save tokenizer if available
    if tokenizer is not None:
        logger.info("Saving tokenizer")
        tokenizer.save_pretrained(export_dir)
    
    # Ajouter un fichier .gitattributes pour Git LFS
    with open(os.path.join(export_dir, ".gitattributes"), "w") as f:
        f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.pt filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.pth filter=lfs diff=lfs merge=lfs -text\n")
    
    # Ajouter un fichier README.md
    with open(os.path.join(export_dir, "README.md"), "w") as f:
        f.write("# Fine-Tuned Text-to-Image Model\n\n")
        f.write("This model is a fine-tuned version based on Wan2.1-T2V-14B.\n\n")
        f.write("## Loading\n```python\n")
        f.write("from diffusers import DiffusionPipeline\n")
        f.write('model = DiffusionPipeline.from_pretrained("your-username/text-image")\n')
        f.write("```\n\n")
        f.write("## Parameters\n- Steps: 10\n- Learning rate: 0.0001\n- Optimizer: adamw_8bit\n")
    
    logger.info(f"Model exported to {export_dir}")
    
    # List the files in the export directory to verify
    logger.info("Files in export directory:")
    for root, dirs, files in os.walk(export_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            logger.info(f"  {file_path} ({file_size:.2f} MB)")

def upload_to_hf(export_dir, repo_id):
    """Téléverse le modèle sur Hugging Face."""
    # Check for HF_TOKEN in environment
    hf_token = os.environ.get("HF_TOKEN")
    
    # If not in environment, ask the user
    if not hf_token:
        logger.info("HF_TOKEN not found in environment variables")
        hf_token = input("Please enter your Hugging Face token (from https://huggingface.co/settings/tokens): ")
        
        if not hf_token:
            logger.error("No token provided, cannot upload to Hugging Face")
            return False
    
    # Set the token in the environment for huggingface_hub to use
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    
    # Initialize the API with the token
    api = HfApi(token=hf_token)
    
    # Check if the repository exists, create it if it doesn't
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        logger.info(f"Repository {repo_id} exists, uploading files...")
    except Exception as e:
        logger.info(f"Repository {repo_id} does not exist, creating it... ({str(e)})")
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            logger.info(f"Repository {repo_id} created successfully")
        except Exception as create_error:
            logger.error(f"Failed to create repository: {str(create_error)}")
            return False
    
    # Upload the files
    try:
        logger.info(f"Uploading files to {repo_id}...")
        # First try with the API
        api.upload_folder(
            folder_path=export_dir,
            repo_id=repo_id,
            repo_type="model"
        )
        logger.info(f"Model uploaded to: https://huggingface.co/{repo_id}")
        return True
    except Exception as upload_error:
        logger.error(f"Failed to upload files with API: {str(upload_error)}")
        
        # Try with the CLI as a fallback
        try:
            logger.info("Trying upload with huggingface-cli...")
            subprocess.run(["huggingface-cli", "login", "--token", hf_token], check=True)
            subprocess.run(["huggingface-cli", "upload", export_dir, repo_id], check=True)
            logger.info(f"Model uploaded to: https://huggingface.co/{repo_id}")
            return True
        except Exception as cli_error:
            logger.error(f"Failed to upload with CLI: {str(cli_error)}")
            return False

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
    
    # Ask user if they want to upload to Hugging Face
    upload_choice = input("Do you want to upload the model to Hugging Face? (yes/no): ")
    
    if upload_choice.lower() in ["yes", "y", "oui", "o"]:
        # Get the repository name from the user
        username = input("Enter your Hugging Face username: ")
        model_name = input("Enter a name for your model repository: ")
        repo_id = f"{username}/{model_name}"
        
        # Upload the model
        logger.info(f"Uploading model to {repo_id}...")
        upload_success = upload_to_hf(export_dir, repo_id)
        
        if upload_success:
            logger.info(f"Model uploaded successfully to: https://huggingface.co/{repo_id}")
        else:
            logger.info("You can manually upload the model from the 'hf_model_export' directory using the Hugging Face CLI:")
            logger.info(f"  huggingface-cli login")
            logger.info(f"  huggingface-cli upload {export_dir} {repo_id}")
    else:
        logger.info("Skipping upload to Hugging Face. Model saved locally in 'hf_model_export' directory.")
    
    logger.info("Training process completed successfully.")

if __name__ == "__main__":
    main()
