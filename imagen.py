import requests
import torch
import numpy as np
from PIL import Image
import base64
import io
import os

from google import genai
from google.genai import types

class ImagenT2ICallerNode:
    """
    A custom node that takes a prompt, aspect ratio, and number of images,
    calls an external API to generate images, and returns them.
    """
    # Define the input types for your node
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
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
            },
        }

    # Define the return types of your node
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    # Define the function that will be executed when the node is called
    FUNCTION = "call_image_api"

    # Define the category for your node
    CATEGORY = "API"

    def __init__(self):
        PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT", "vertex-generative-media"))
        LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    def call_image_api(self, prompt, model, num_images, aspect_ratio):
        """
        This function is called when the node is executed.
        It sends the parameters to the specified API URL and returns the generated images.
        """
        
        api_response = self.client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                aspect_ratio=aspect_ratio,
                number_of_images=num_images,
            ),
        )
        
        if not len(api_response.generated_images):
            print("API did not return any images.")
            # Return a black dummy image tensor to avoid crashing the workflow
            return (torch.zeros(1, 64, 64, 3, dtype=torch.float32),)

        image_tensors = []
        for image in api_response.generated_images:
            pil_image = image.image._pil_image.convert("RGBA")
            image_array = np.array(pil_image).astype(np.float32) / 255.0
            
            # Convert NumPy array to a PyTorch tensor (H, W, C) -> (1, H, W, C)
            image_tensor = torch.from_numpy(image_array)[None,]
            image_tensors.append(image_tensor)

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
