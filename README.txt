Prompt-Driven Background Generation for Image Augmentation

This script enables automatic captioning, refinement, and background generation for images. 
It extracts the foreground object, inpaints the background, generates a meaningful caption from the clean background, and uses either a local language model or diffusion model to generate or retrieve similar backgrounds.

Features-

1. Foreground-background separation using rembg

2. Inpainting to restore clean background

3. Image captioning using Microsoft GIT (microsoft/git-large-coco)

4. Prompt refinement using GPT4All (mistral-7b-instruct)

5. Background generation using Stable Diffusion (runwayml/stable-diffusion-v1-5)

How It Works-

1. Foreground Removal-

    Uses rembg to separate foreground from background.

2. Inpainting-

    Inpaints the removed region using OpenCV to create a natural background.

3. Captioning-

    Passes the cleaned background to Microsoft’s git-large-coco model for image captioning.

4. Prompt Refinement-

    Refines the generated caption using a local GGUF model loaded via GPT4All.

5. Background Generation-

    Uses Stable Diffusion to generate synthetic backgrounds from the refined prompt.


Requirements-

Install all required dependencies:

    pip install torch torchvision transformers rembg diffusers accelerate opencv-python pillow

For local LLM support, ensure to install gpt4all
    
    pip install gpt4all

Download links for models used in the repo-

gpt4all model - https://gpt4all.io/models/gguf/mistral-7b-instruct-v0.1.Q4_0.gguf

