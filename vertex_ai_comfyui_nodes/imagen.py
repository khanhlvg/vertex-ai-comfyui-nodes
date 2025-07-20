import requests
import torch
import numpy as np
from PIL import Image
import base64
import io
import os
import random
import asyncio

from google import genai
from google.genai import types

from .utils import tensor_to_pil, pil_to_base64, base64_to_tensor

class ImagenT2ICallerNode:
    """
    A custom node that calls the Imagen API to generate images from a prompt.
    """
    # Define the input types for your node
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Imagen node.
        """
        return {
            "required": {
                "project_id": ("STRING", {
                    "multiline": False,
                    "default": os.environ.get("GOOGLE_CLOUD_PROJECT")
                }),
                "location": ("STRING", {
                    "multiline": False,
                    "default": os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A majestic lion in the savannah, 4k, high detail"
                }),
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"],),
                "model": ([
                    "imagen-4.0-generate-preview-06-06", 
                    "imagen-4.0-ultra-generate-preview-06-06", 
                    "imagen-4.0-fast-generate-preview-06-06", 
                    ],),
                "seed": ("INT", {
                    "default": random.randint(0, 4294967295),
                    "min": 0,
                    "max": 4294967295
                }),
                "safety_filter_level": (["BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE", "BLOCK_NONE"],),
                "person_generation": (["ALLOW_ALL", "ALLOW_ADULT", "DONT_ALLOW"],),
                "enhancePrompt": ("BOOLEAN", {"default": True}),
            },
        }

    # Define the return types of your node
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    # Define the function that will be executed when the node is called
    FUNCTION = "call_image_api"

    # Define the category for your node
    CATEGORY = "Vertex AI"

    def __init__(self):
        self.client = None

    async def call_image_api(self, project_id, location, prompt, model, num_images, aspect_ratio, seed, safety_filter_level, person_generation, enhancePrompt):
        """
        This function is called when the node is executed.
        It sends the parameters to the specified API URL and returns the generated images.
        """
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

        api_response = await asyncio.to_thread(
            self.client.models.generate_images,
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                aspect_ratio=aspect_ratio,
                number_of_images=num_images,
                seed=seed,
                safety_filter_level=safety_filter_level,
                person_generation=person_generation,
                enhance_prompt=enhancePrompt,
                add_watermark=False,
            ),
        )
        
        if not len(api_response.generated_images):
            print("API did not return any images.")
            # Return a black dummy image tensor to avoid crashing the workflow
            return (torch.zeros(1, 64, 64, 3, dtype=torch.float32),)

        image_tensors = []
        for image in api_response.generated_images:
            pil_image = image.image._pil_image.convert("RGBA")
            image_tensors.append(base64_to_tensor(pil_to_base64(pil_image)))

        # Stack all tensors into a single batch tensor
        batch_tensor = torch.cat(image_tensors, 0)

        return (batch_tensor,)

# A dictionary that ComfyUI uses to register the nodes in this file
NODE_CLASS_MAPPINGS = {
    "Imagen_T2I": ImagenT2ICallerNode
}

# A dictionary that ComfyUI uses to display the node names in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "Imagen_T2I": "Imagen Text-to-Image"
}