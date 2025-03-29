import os
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import FluxPipeline
from peft import LoraConfig, get_peft_model
from PIL import Image
import torchvision.transforms as transforms
import logging
from accelerate import Accelerator
import random
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TrainingConfig:
    model_name = "black-forest-labs/FLUX.1-dev"
    dataset_dir = "./datasets/my_dataset"
    output_dir = "./output/models"
    resolution = 1024
    batch_size = 6
    gradient_accumulation_steps = 2
    learning_rate = 2e-5
    max_train_steps = 3000
    checkpointing_steps = 300
    validation_steps = 300
    validation_prompt = "a serene landscape with mountains and a lake"
    caption_dropout_prob = 0.1
    lr_scheduler = "cosine"
    lr_warmup_steps = 100
    seed = 42
    mixed_precision = "bf16"
    num_inference_steps = 20
    early_stopping_patience = 3
    max_checkpoints = 3
    num_timesteps = 28
    hf_token = "your_hugging_face_token_here"  # Replace with your actual token

class ImageCaptionDataset(Dataset):
    def __init__(self, dataset_dir, resolution, tokenizer, text_encoder, caption_dropout_prob=0.1):
        self.dataset_dir = dataset_dir
        self.resolution = resolution
        self.caption_dropout_prob = caption_dropout_prob
        self.image_files = [f for f in os.listdir(dataset_dir) if f.endswith((".jpg", ".png"))]
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.text_embeds = {}
        with torch.no_grad():
            for img_file in self.image_files:
                caption_path = os.path.join(dataset_dir, img_file.rsplit(".", 1)[0] + ".txt")
                try:
                    with open(caption_path, "r") as f:
                        caption = f.read().strip()
                except FileNotFoundError:
                    caption = ""
                raw_tokens = tokenizer(caption, return_tensors="pt").input_ids
                if raw_tokens.shape[1] > 77:
                    logger.warning(f"Caption for {img_file} exceeds 77 tokens ({raw_tokens.shape[1]}). Truncating...")
                    with open("long_captions.txt", "a") as f:
                        f.write(f"{img_file}: {caption} (tokens: {raw_tokens.shape[1]})\n")
                inputs = tokenizer(
                    caption,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                )
                self.text_embeds[img_file] = text_encoder(inputs.input_ids.to(text_encoder.device))[0].cpu()

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        caption_embed = self.text_embeds[self.image_files[idx]]
        if random.random() < self.caption_dropout_prob:
            caption_embed = torch.zeros_like(caption_embed)
        return {"pixel_values": image, "text_embeds": caption_embed}

    def __len__(self):
        return len(self.image_files)

def train_lora(config):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir="./logs"
    )
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    try:
        logger.info("Loading FLUX.1-dev model...")
        pipeline = FluxPipeline.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            token=config.hf_token
        )
        transformer = pipeline.transformer
        vae = pipeline.vae
        text_encoder = pipeline.text_encoder

        logger.info("Preparing models for multi-GPU...")
        transformer, vae, text_encoder = accelerator.prepare(transformer, vae, text_encoder)

        logger.info(f"Number of devices: {accelerator.num_processes}")
        logger.info(f"Device for transformer: {next(transformer.parameters()).device}")
        logger.info(f"Device for VAE: {next(vae.parameters()).device}")
        logger.info(f"Device for text encoder: {next(text_encoder.parameters()).device}")

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "to_q", "to_k", "to_v",
                "attn.to_out.0",
                "add_q_proj", "add_k_proj", "add_v_proj",
                "to_add_out"
            ],
            lora_dropout=0.05
        )
        
        logger.info("Applying LoRA to transformer...")
        transformer = get_peft_model(transformer, lora_config)
        logger.info(f"LoRA applied. Trainable parameters: {transformer.print_trainable_parameters()}")

        dataset = ImageCaptionDataset(
            config.dataset_dir,
            config.resolution,
            pipeline.tokenizer,
            text_encoder,
            config.caption_dropout_prob
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )

        optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_train_steps - config.lr_warmup_steps
        )

        transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, dataloader, lr_scheduler
        )

        alphas_cumprod = torch.linspace(0.99, 0.1, config.num_timesteps).to(accelerator.device)
        logger.info("Starting training...")
        global_step = 0
        os.makedirs(config.output_dir, exist_ok=True)
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=torch.float16)
                text_embeds = batch["text_embeds"].to(dtype=torch.float16)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, config.num_timesteps - 4, (latents.shape[0],), device=accelerator.device)
                alpha = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                sigma = (1 - alpha).sqrt()
                noisy_latents = alpha * latents + sigma * noise
                pred_noise = transformer(noisy_latents, timesteps, encoder_hidden_states=text_embeds).sample
                loss = torch.nn.functional.mse_loss(pred_noise, noise)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            global_step += 1
            if global_step % 10 == 0:
                logger.info(f"Step {global_step}/{config.max_train_steps}, Loss: {loss.item():.4f}")
            if global_step >= 50:  # Short run to test multi-GPU
                break
        logger.info("Training test completed!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        torch.cuda.empty_cache()
        raise
    finally:
        torch.cuda.empty_cache()
        accelerator.end_training()

# Run in Jupyter
config = TrainingConfig()
train_lora(config)
