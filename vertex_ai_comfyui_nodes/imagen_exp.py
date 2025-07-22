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
import numpy as np
from PIL import Image
import base64
import io
from google.cloud import aiplatform
import asyncio

from .utils import tensor_to_pil, pil_to_base64, base64_to_tensor

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

    async def generate_contextualized_image(self, project_id, location, prompt, image1, image2=None, image3=None, productDescription=None, sampleCount=1, seed=0, safetySetting="block_low_and_above", personGeneration="allow_adult"):
        """
        Generates a new image by placing the input product(s) in a new context.

        This asynchronous method initializes the AI Platform client, prepares the
        input images and parameters, and calls the prediction endpoint for the
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
        aiplatform.init(project=project_id, location=location)
        api_regional_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_regional_endpoint}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        # Collect and process all provided images.
        images = [image1, image2, image3]
        image_bytes_list = []
        for img_tensor in images:
            if img_tensor is not None:
                pil_img = tensor_to_pil(img_tensor)
                base64_img = pil_to_base64(pil_img)
                image_bytes_list.append(base64_img)

        # Construct the instance payload for the API request.
        instances = []
        instance = {"productImages": []}
        for product_image_bytes in image_bytes_list:
            product_image = {"image": {"bytesBase64Encoded": product_image_bytes}}
            instance["productImages"].append(product_image)

        if productDescription:
            instance["productImages"][0]["productConfig"] = {
                "productDescription": productDescription
            }

        if prompt:
            instance["prompt"] = prompt

        # Set the generation parameters.
        parameters = {"sampleCount": sampleCount, "seed": seed}
        if safetySetting:
            parameters["safetySetting"] = safetySetting
        if personGeneration:
            parameters["personGeneration"] = personGeneration

        instances.append(instance)

        # Define the model endpoint and make the prediction call.
        model_endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/imagen-product-recontext-preview-06-30"
        response = await asyncio.to_thread(client.predict, endpoint=model_endpoint, instances=instances, parameters=parameters)

        # Process the response and convert the base64 images to tensors.
        image_tensors = []
        for prediction in response.predictions:
            base64_image = prediction["bytesBase64Encoded"]
            image_tensors.append(base64_to_tensor(base64_image))

        # If no images are returned, provide a dummy tensor.
        if not image_tensors:
            print("API did not return any images.")
            return (torch.zeros(1, 64, 64, 3, dtype=torch.float32),)

        # Combine the individual tensors into a single batch and return.
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

    async def virtual_try_on(self, project_id, location, person_image, product_image, sampleCount=1, seed=0, safetySetting="block_low_and_above", personGeneration="allow_adult"):
        """
        Performs the virtual try-on by combining a person and a product image.

        This asynchronous method sets up the AI Platform client, processes the
        input person and product images, and calls the prediction endpoint for the
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
        aiplatform.init(project=project_id, location=location)
        api_regional_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_regional_endpoint}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        # Convert the input tensors to PIL images and then to base64 strings.
        person_pil = tensor_to_pil(person_image)
        product_pil = tensor_to_pil(product_image)

        person_b64 = pil_to_base64(person_pil)
        product_b64 = pil_to_base64(product_pil)

        # Construct the instance payload for the API request.
        instances = [
            {
                "personImage": {"image": {"bytesBase64Encoded": person_b64}},
                "productImages": [{"image": {"bytesBase64Encoded": product_b64}}],
            }
        ]

        # Set the generation parameters.
        parameters = {"sampleCount": sampleCount, "seed": seed}
        if safetySetting:
            parameters["safetySetting"] = safetySetting
        if personGeneration:
            parameters["personGeneration"] = personGeneration

        # Define the model endpoint and make the prediction call.
        model_endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/virtual-try-on-exp-05-31"
        response = await asyncio.to_thread(client.predict, endpoint=model_endpoint, instances=instances, parameters=parameters)

        # Process the response and convert the base64 images to tensors.
        image_tensors = []
        for prediction in response.predictions:
            base64_image = prediction["bytesBase64Encoded"]
            image_tensors.append(base64_to_tensor(base64_image))

        # If no images are returned, provide a dummy tensor.
        if not image_tensors:
            print("API did not return any images.")
            return (torch.zeros(1, 64, 64, 3, dtype=torch.float32),)

        # Combine the individual tensors into a single batch and return.
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
