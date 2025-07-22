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
    A custom node that takes product images and a prompt to generate a new image with the product in a different context.
    """
    @classmethod
    def INPUT_TYPES(s):
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
        aiplatform.init(project=project_id, location=location)
        api_regional_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_regional_endpoint}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        images = [image1, image2, image3]
        image_bytes_list = []
        for img_tensor in images:
            if img_tensor is not None:
                pil_img = tensor_to_pil(img_tensor)
                base64_img = pil_to_base64(pil_img)
                image_bytes_list.append(base64_img)

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

        parameters = {"sampleCount": sampleCount, "seed": seed}
        if safetySetting:
            parameters["safetySetting"] = safetySetting
        if personGeneration:
            parameters["personGeneration"] = personGeneration

        instances.append(instance)

        model_endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/imagen-product-recontext-preview-06-30"
        response = await asyncio.to_thread(client.predict, endpoint=model_endpoint, instances=instances, parameters=parameters)

        image_tensors = []
        for prediction in response.predictions:
            base64_image = prediction["bytesBase64Encoded"]
            image_tensors.append(base64_to_tensor(base64_image))

        if not image_tensors:
            print("API did not return any images.")
            return (torch.zeros(1, 64, 64, 3, dtype=torch.float32),)

        batch_tensor = torch.cat(image_tensors, 0)
        return (batch_tensor,)

class VirtualTryOnNode:
    """
    A custom node that takes a person image and a product image to generate a new image with the person wearing the product.
    """
    @classmethod
    def INPUT_TYPES(s):
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
        aiplatform.init(project=project_id, location=location)
        api_regional_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_regional_endpoint}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        person_pil = tensor_to_pil(person_image)
        product_pil = tensor_to_pil(product_image)

        person_b64 = pil_to_base64(person_pil)
        product_b64 = pil_to_base64(product_pil)

        instances = [
            {
                "personImage": {"image": {"bytesBase64Encoded": person_b64}},
                "productImages": [{"image": {"bytesBase64Encoded": product_b64}}],
            }
        ]

        parameters = {"sampleCount": sampleCount, "seed": seed}
        if safetySetting:
            parameters["safetySetting"] = safetySetting
        if personGeneration:
            parameters["personGeneration"] = personGeneration

        model_endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/virtual-try-on-exp-05-31"
        response = await asyncio.to_thread(client.predict, endpoint=model_endpoint, instances=instances, parameters=parameters)

        image_tensors = []
        for prediction in response.predictions:
            base64_image = prediction["bytesBase64Encoded"]
            image_tensors.append(base64_to_tensor(base64_image))

        if not image_tensors:
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