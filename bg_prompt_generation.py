from transformers import AutoProcessor,AutoModelForCausalLM
from PIL import Image
import torch
from rembg import remove
import io
import cv2
import numpy as np
from bg_web_scraping import download_backgrounds

input_path = r"C:\Users\hrish\Desktop\datasets\dataset_004\input\images\Movie-on-2-18-25-at-8_25-PM_mov-0003_jpg.rf.40ae588a0c9e10795fd345526589e35a.jpg"
bg_clean_path = r"C:\Users\hrish\Desktop\datasets\dataset_004\input\bg_only.png"

# --- Background Extraction ---
original = Image.open(input_path).convert("RGB")

with open(input_path, 'rb') as inp_file:
    input_bytes = inp_file.read()
    fg_removed = remove(input_bytes)

fg_image = Image.open(io.BytesIO(fg_removed)).convert("RGBA")

# Generate mask
mask = np.array(fg_image.split()[-1])
mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

# Inpaint
original_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
inpainted = cv2.inpaint(original_cv, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
bg_clean = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
bg_clean.save(bg_clean_path)

# --- Caption Generation ---
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")

# Load and preprocess the image
image = Image.open(bg_clean_path)
inputs = processor(images=image, return_tensors="pt")

# Generate the caption
generated_ids = model.generate(pixel_values=inputs["pixel_values"], max_length=50)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(caption)

from gpt4all import GPT4All

model_name = "mistral-7b-instruct-v0.1.Q4_0.gguf"
model_path = "C:/Users/hrish/Desktop/new_repo"

model = GPT4All(model_name=model_name,model_path=model_path, allow_download=False)

response = model.generate(f"Refine the prompt: {caption}")
print("Response:", response)


# download_backgrounds(caption,10,r"C:\Users\hrish\Desktop\datasets\dataset_004\backgrounds\web_scraping")

from diffusers import StableDiffusionPipeline
import torch

# Load the pipeline for CPU
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cpu")

# Generate image
# prompt = "a futuristic cityscape at sunset"
image = pipe(caption, num_inference_steps=30, guidance_scale=7.5).images[0]

# Save image
image.save("output2.png")
print("Image saved as output.png")
