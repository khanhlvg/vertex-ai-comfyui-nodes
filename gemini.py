import os
import torch
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from google.genai.types import Part
import io

class GeminiCallerNode:
    """
    A custom node that calls the Gemini API to generate text from a prompt and optional images.
    """
    def __init__(self):
        self.client = None

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Gemini node.
        """
        return {
            "required": {
                "project_id": ("STRING", {
                    "multiline": False,
                    "default": os.environ.get("GOOGLE_CLOUD_PROJECT")
                }),
                "region": ("STRING", {
                    "multiline": False,
                    "default": os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
                }),
                "model_name": ([
                    'gemini-2.5-flash',
                    'gemini-2.5-pro',
                    'gemini-2.5-flash-lite-preview-06-17',                    
                    'gemini-2.0-flash',
                    'gemini-2.0-flash-lite',
                ],),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "How are you doing today?",
                }),
            },
            "optional": {
                "system_instruction": ("STRING", {
                    "multiline": True,
                }),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"

    CATEGORY = "Vertex AI"

    def tensor_to_pil(self, tensor):
        """
        Converts a PyTorch tensor to a PIL Image.
        """
        if tensor is None:
            return None
        # Convert tensor to numpy array
        image_np = tensor.squeeze().cpu().numpy()
        # Denormalize if necessary (assuming tensor is in [0, 1] range)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        # Convert to PIL Image
        return Image.fromarray(image_np)

    def generate_text(self, project_id, region, model_name, prompt, system_instruction=None, image1=None, image2=None, image3=None):
        """
        Generates text using the Gemini API based on a prompt and optional images.
        """
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=region)
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        ) if system_instruction else None

        contents = [prompt]
        images = [image1, image2, image3]
        for img_tensor in images:
            if img_tensor is not None:
                pil_img = self.tensor_to_pil(img_tensor)
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                contents.append(Part.from_bytes(data=img_bytes, mime_type="image/png"))

        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        return (response.text,)

# A dictionary that ComfyUI uses to register the nodes in this file
NODE_CLASS_MAPPINGS = {
    "Gemini": GeminiCallerNode
}

# A dictionary that ComfyUI uses to display the node names in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini": "Gemini"
}