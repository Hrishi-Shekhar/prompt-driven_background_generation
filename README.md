
# 🧠 Prompt-Driven Background Generation for Image Augmentation

This repository contains a modular, production-ready pipeline that automates **background generation** for images using a combination of:

- Foreground removal (rembg)  
- Background inpainting (OpenCV)  
- Image captioning (Microsoft GIT)  
- Prompt refinement (GPT4All with Mistral)  
- Background generation (Stable Diffusion)

---

## 🚀 Pipeline Overview

This pipeline allows you to take an input image and generate a **new, realistic background** aligned with the scene's context, making it ideal for **image augmentation**, especially in medical or domain-sensitive applications.

### 🔧 What the pipeline does:

1. **Foreground Removal**  
   Uses `rembg` to cleanly extract the object from the image.

2. **Inpainting**  
   Uses OpenCV's inpainting to restore the background after object removal.

3. **Caption Generation**  
   Applies Microsoft’s `git-large-coco` model to describe the clean background.

4. **Prompt Refinement**  
   Uses a **local GPT4All model** (e.g., `mistral-7b-instruct`) to enhance the generated caption.

5. **Background Generation**  
   Uses Stable Diffusion (`runwayml/stable-diffusion-v1-5`) to generate a high-quality background from the refined prompt.

---

## 🗂️ Folder Structure

```
prompt-driven-background-generation/
├── bg_prompt_generation.py                                # Main pipeline script
├── pipeline.log                             # Auto-generated logs
├── Ancylostoma-Spp--68-_jpg...jpg           # Sample input image
├── bg_only.png                              # Clean background (after inpainting)
├── generated_image.png                      # Final generated background image
```

---

## 📥 Requirements

Install the dependencies:

```bash
pip install torch torchvision transformers rembg diffusers accelerate opencv-python pillow gpt4all
```

> ⚠️ Make sure you are using a supported version of Python (>=3.8).

---

## 📁 Model Downloads

Download the following model manually before running the pipeline:

- 🔹 **GPT4All (Mistral GGUF)**  
  Download from: https://gpt4all.io/models/gguf/mistral-7b-instruct-v0.1.Q4_0.gguf  
  Place it inside a local `models/` directory or update the `model_path` in the script accordingly.

- 🔹 **Stable Diffusion**  
  Automatically downloaded via `diffusers` from Hugging Face (`runwayml/stable-diffusion-v1-5`) on first use.

- 🔹 **Microsoft GIT Model**  
  Also downloaded automatically: `microsoft/git-large-coco`

---

## ▶️ How to Run

1. Place your **input image** (e.g., a medical or object image) in the project directory.

2. Update these variables in `bg_prompt_generation.py`:

```python
input_path = "your_input_image.jpg"
model_name = "mistral-7b-instruct-v0.1.Q4_0.gguf"
model_path = "models/"
```

3. Run the script:

```bash
python bg_prompt_generation.py
```

4. The pipeline will output:

- `bg_only.png` – inpainted clean background  
- `generated_image.png` – AI-generated semantic background

---

## 🛠️ Customization

- To use your own GPT4All model, just change `model_name` and `model_path`.
- You can replace the image with any relevant domain image.
- Supports CPU and CUDA (automatically detected).

---

## 🧪 Example Use Case

With `Ancylostoma-Spp--68-...jpg`, the pipeline might:

1. Extract the worm foreground.
2. Inpaint the lab background.
3. Generate a caption: _“a clinical laboratory background with a sterile table”_.
4. Refine to: _“a high-resolution clinical background suitable for microscopic worm imaging”_.
5. Generate a synthetic, context-aware background image.

---

## 📌 Notes

- Internet is required to fetch Hugging Face models unless cached locally.
- Ensure your system meets the VRAM requirements for Stable Diffusion (8GB+ recommended).
- Logs are written to `pipeline.log` for debugging.

---


