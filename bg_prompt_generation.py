import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import io
import torch
import logging
from logging.handlers import RotatingFileHandler
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from transformers import AutoProcessor, AutoModelForCausalLM
from gpt4all import GPT4All
from diffusers import StableDiffusionPipeline

# --- Enhanced Logging Setup ---
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    fh = RotatingFileHandler(
        'pipeline.log', 
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    fh.setLevel(logging.INFO)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"


# 1. Background Extraction -----------------------------------------------------
def extract_background(input_path, output_path):
    logger.info("Starting background extraction...")
    original = Image.open(input_path).convert("RGB")

    with open(input_path, 'rb') as inp_file:
        input_bytes = inp_file.read()
        fg_removed = remove(input_bytes)

    fg_image = Image.open(io.BytesIO(fg_removed)).convert("RGBA")
    mask = cv2.threshold(np.array(fg_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)[1]
    original_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(original_cv, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    bg_clean = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    bg_clean.save(output_path)
    logger.info(f"Background image saved to: {output_path}")
    return output_path


# 2. Caption Generation -------------------------------------------------------
def generate_caption(image_path):
    logger.info("Generating caption with GIT model...")
    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco").to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values=inputs["pixel_values"], max_length=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.info(f"Generated Caption: {caption}")
    return caption


# 3. Prompt Refinement --------------------------------------------------------
def refine_prompt(prompt, model_name, model_path):
    logger.info("Refining prompt with GPT4All...")
    model = GPT4All(model_name=model_name, model_path=model_path, allow_download=False)
    refined = model.generate(f"Refine the prompt: {prompt}")
    logger.info(f"Refined Prompt: {refined}")
    return refined


# 4. Image Generation ---------------------------------------------------------
def generate_image(prompt, output_path="generated_image.png"):
    logger.info("Generating image using Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cpu")

    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(output_path)
    logger.info(f"Image saved as {output_path}")


# --- Main Execution ----------------------------------------------------------
if __name__ == "__main__":
    input_path = r"C:\Users\hrish\Desktop\datasets\dataset_001\input\images\Ancylostoma-Spp--68-_jpg.rf.92487af38bb34993d2716b49791b7866.jpg"
    bg_clean_path = "bg_only.png"
    model_name = "mistral-7b-instruct-v0.1.Q4_0.gguf"
    model_path = r"C:\Users\hrish\Desktop\prompt-driven_background_generation"

    try:
        logger.info("Starting pipeline execution")
        clean_bg_path = extract_background(input_path, bg_clean_path)
        caption = generate_caption(clean_bg_path)
        refined_prompt = refine_prompt(caption, model_name, model_path)
        generate_image(refined_prompt)
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)