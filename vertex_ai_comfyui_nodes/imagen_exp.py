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

import os
import random
import torch
import asyncio

from google import genai
from google.genai import types

from .utils import pil_to_base64, base64_to_tensor, tensor_to_temp_image_file

class ImagenProductRecontextNode:
    """
    A ComfyUI node for re-contextualizing product images using Imagen.

    This node takes one or more product images and a text prompt to generate a
    new image that places the product in a different setting or context, as
    described by the prompt.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Imagen Product Re-contextualization node.

        This includes required inputs like project ID, location, prompt, and at
        least one image. Optional inputs allow for more images, a product
        description, and control over generation parameters.
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
                "model": (["imagen-product-recontext-preview-06-30"],),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A product photo of a car on a race track."
                }),
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "productDescription": ("STRING", {
                    "multiline": True,
                }),
                "sampleCount": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": random.randint(0, 4294967295),
                    "min": 0,
                    "max": 4294967295
                }),
                "safetySetting": (["BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE", "BLOCK_NONE"],),
                "personGeneration": (["ALLOW_ALL", "ALLOW_ADULT", "DONT_ALLOW"],)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "generate_contextualized_image"

    CATEGORY = "Vertex AI"

    def __init__(self):
        self.client = None

    async def generate_contextualized_image(self, project_id, location, model, prompt, image1, image2=None, image3=None, productDescription=None, sampleCount=1, seed=0, safetySetting="BLOCK_LOW_AND_ABOVE", personGeneration="ALLOW_ADULT"):
        """
        Generates a new image by placing the input product(s) in a new context.

        This asynchronous method initializes the genai.Client, prepares the
        input images and parameters, and calls the recontext_image method for the
        Imagen product re-contextualization model. The resulting images are
        returned as a tensor batch.

        Args:
            project_id (str): The Google Cloud project ID.
            location (str): The Google Cloud region.
            prompt (str): The description of the new context.
            image1 (torch.Tensor): The primary product image.
            image2 (torch.Tensor, optional): A second product image.
            image3 (torch.Tensor, optional): A third product image.
            productDescription (str, optional): A description of the product.
            sampleCount (int): The number of images to generate.
            seed (int): The random seed for generation.
            safetySetting (str): The safety filtering level.
            personGeneration (str): The setting for person generation.

        Returns:
            tuple: A tuple containing a batch of generated images as a torch.Tensor.
        """
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

        images = [image1, image2, image3]
        product_images = []
        temp_files = []
        for img_tensor in images:
            if img_tensor is not None:
                temp_path = tensor_to_temp_image_file(img_tensor)
                product_images.append(types.ProductImage(product_image=types.Image.from_file(location=temp_path)))
                temp_files.append(temp_path)

        source = types.RecontextImageSource(
            prompt=prompt,
            product_images=product_images,
        )

        config = types.RecontextImageConfig(
            number_of_images=sampleCount,
            safety_filter_level=safetySetting,
            person_generation=personGeneration,
            seed=seed,
            add_watermark=False
        )

        response = await asyncio.to_thread(
            self.client.models.recontext_image,
            model=model,
            source=source,
            config=config,
        )

        for temp_path in temp_files:
            os.remove(temp_path)

        image_tensors = []
        for image in response.generated_images:
            if not image.image.image_bytes:
                print("No image returned by the API. Your request was likely blocked by the safety filters.")
                continue
            try:
                pil_image = image.image._pil_image.convert("RGBA")
                image_tensors.append(base64_to_tensor(pil_to_base64(pil_image)))
            except (ValueError, AttributeError):
                print("Skipping an image that could not be decoded.")
                continue

        if not image_tensors:
            # If no images are returned, provide a dummy tensor.
            print("API did not return any images.")
            return (torch.zeros(1, 64, 64, 3, dtype=torch.float32),)

        batch_tensor = torch.cat(image_tensors, 0)
        return (batch_tensor,)

class VirtualTryOnNode:
    """
    A ComfyUI node for the Virtual Try-On feature using Imagen.

    This node takes an image of a person and a product image (e.g., clothing)
    and generates a new image showing the person wearing the product.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Virtual Try-On node.

        This includes required inputs for the project ID, location, a person image,
        and a product image. Optional parameters are available for controlling the
        number of output images, seed, and safety settings.
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
                "model": (["virtual-try-on-preview-08-04"],),
                "person_image": ("IMAGE",),
                "product_image": ("IMAGE",),
            },
            "optional": {
                "sampleCount": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": random.randint(0, 4294967295),
                    "min": 0,
                    "max": 4294967295
                }),
                "safetySetting": (["BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE", "BLOCK_NONE"],),
                "personGeneration": (["ALLOW_ALL", "ALLOW_ADULT"],)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "virtual_try_on"

    CATEGORY = "Vertex AI"

    def __init__(self):
        self.client = None

    async def virtual_try_on(self, project_id, location, model, person_image, product_image, sampleCount=1, seed=0, safetySetting="BLOCK_LOW_AND_ABOVE", personGeneration="ALLOW_ADULT"):
        """
        Performs the virtual try-on by combining a person and a product image.

        This asynchronous method sets up the genai.Client, processes the
        input person and product images, and calls the recontext_image method for the
        virtual try-on model. The generated images are returned as a tensor batch.

        Args:
            project_id (str): The Google Cloud project ID.
            location (str): The Google Cloud region.
            person_image (torch.Tensor): The image of the person.
            product_image (torch.Tensor): The image of the product.
            sampleCount (int): The number of images to generate.
            seed (int): The random seed for generation.
            safetySetting (str): The safety filtering level.
            personGeneration (str): The setting for person generation.

        Returns:
            tuple: A tuple containing a batch of generated images as a torch.Tensor.
        """
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

        person_image_path = tensor_to_temp_image_file(person_image)
        product_image_path = tensor_to_temp_image_file(product_image)

        source = types.RecontextImageSource(
            person_image=types.Image.from_file(location=person_image_path),
            product_images=[
                types.ProductImage(product_image=types.Image.from_file(location=product_image_path))
            ],
        )

        config = types.RecontextImageConfig(
            number_of_images=sampleCount,
            safety_filter_level=safetySetting,
            person_generation=personGeneration,
            seed=seed,
            add_watermark=False
        )

        response = await asyncio.to_thread(
            self.client.models.recontext_image,
            model=model,
            source=source,
            config=config,
        )

        os.remove(person_image_path)
        os.remove(product_image_path)

        image_tensors = []
        for image in response.generated_images:
            if not image.image.image_bytes:
                print("No image returned by the API. Your request was likely blocked by the safety filters.")
                continue
            try:
                pil_image = image.image._pil_image.convert("RGBA")
                image_tensors.append(base64_to_tensor(pil_to_base64(pil_image)))
            except (ValueError, AttributeError):
                print("Skipping an image that could not be decoded.")
                continue

        if not image_tensors:
            # If no images are returned, provide a dummy tensor.
            print("API did not return any images.")
            return (torch.zeros(1, 64, 64, 3, dtype=torch.float32),)

        batch_tensor = torch.cat(image_tensors, 0)
        return (batch_tensor,)

NODE_CLASS_MAPPINGS = {
    "Imagen_Product_Recontext": ImagenProductRecontextNode,
    "Virtual_Try_On": VirtualTryOnNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Imagen_Product_Recontext": "Imagen Product Recontext",
    "Virtual_Try_On": "Virtual Try-On",
}
