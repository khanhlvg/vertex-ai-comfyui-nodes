# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    A ComfyUI node for generating images from text prompts using the Google Imagen API.

    This node provides a user-friendly interface within ComfyUI to access Imagen's
    text-to-image capabilities. It supports various models, aspect ratios, and advanced
    options like safety filtering and prompt enhancement.
    """
    # Define the input types for your node
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Imagen node.

        This method specifies all the parameters that users can configure in the
        ComfyUI interface, such as the prompt, number of images, model selection,
        and safety settings. Default values are provided for convenience.
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
        """
        Initializes the node by setting the client to None.
        The client will be created on the first execution.
        """
        self.client = None

    async def call_image_api(self, project_id, location, prompt, model, num_images, aspect_ratio, seed, safety_filter_level, person_generation, enhancePrompt):
        """
        This function is called when the node is executed.
        It sends the parameters to the specified API URL and returns the generated images.

        This asynchronous method handles the entire process of calling the Imagen API.
        It initializes the client, constructs the request with all the specified
        parameters, and then processes the response to return the generated images
        as a batch of tensors.

        Args:
            project_id (str): The Google Cloud project ID.
            location (str): The Google Cloud region.
            prompt (str): The text prompt for image generation.
            model (str): The Imagen model to use.
            num_images (int): The number of images to generate.
            aspect_ratio (str): The desired aspect ratio of the images.
            seed (int): The random seed for generation.
            safety_filter_level (str): The safety filtering level.
            person_generation (str): The setting for person generation.
            enhancePrompt (bool): Whether to enhance the prompt.

        Returns:
            tuple: A tuple containing a batch of generated images as a torch.Tensor.
        """
        # Initialize the Imagen client if it hasn't been already.
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

        # Asynchronously call the Imagen API to generate images.
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
        
        # If the API returns no images, print a message and return a dummy tensor.
        if not len(api_response.generated_images):
            print("API did not return any images.")
            # Return a black dummy image tensor to avoid crashing the workflow
            return (torch.zeros(1, 64, 64, 3, dtype=torch.float32),)

        # Process the generated images and convert them to tensors.
        image_tensors = []
        for image in api_response.generated_images:
            # Convert the returned image to a PIL Image in RGBA format.
            pil_image = image.image._pil_image.convert("RGBA")
            # Convert the PIL image to a base64 string and then to a tensor.
            image_tensors.append(base64_to_tensor(pil_to_base64(pil_image)))

        # Stack all individual image tensors into a single batch tensor.
        batch_tensor = torch.cat(image_tensors, 0)

        # Return the batch of images as a tuple.
        return (batch_tensor,)

# A dictionary that ComfyUI uses to register the nodes in this file
NODE_CLASS_MAPPINGS = {
    "Imagen_T2I": ImagenT2ICallerNode
}

# A dictionary that ComfyUI uses to display the node names in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "Imagen_T2I": "Imagen Text-to-Image"
}