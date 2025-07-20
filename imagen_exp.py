import os
import random
import torch
import numpy as np
from PIL import Image
import base64
import io
from google.cloud import aiplatform

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
                "safetySetting": (["BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "BLOCK_NONE"],),
                "personGeneration": (["DONT_ALLOW", "ALLOW_ADULT", "ALLOW_ALL"],)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "generate_contextualized_image"

    CATEGORY = "Vertex AI"

    def tensor_to_pil(self, tensor):
        if tensor is None:
            return None
        image_np = tensor.squeeze().cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        return Image.fromarray(image_np)

    def pil_to_base64(self, pil_image):
        if pil_image is None:
            return None
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def base64_to_tensor(self, base64_image):
        image_data = base64.b64decode(base64_image)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(image_array)[None,]

    def generate_contextualized_image(self, project_id, location, prompt, image1, image2=None, image3=None, productDescription=None, sampleCount=1, seed=0, safetySetting="block_low_and_above", personGeneration="allow_adult"):
        aiplatform.init(project=project_id, location=location)
        api_regional_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_regional_endpoint}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        images = [image1, image2, image3]
        image_bytes_list = []
        for img_tensor in images:
            if img_tensor is not None:
                pil_img = self.tensor_to_pil(img_tensor)
                base64_img = self.pil_to_base64(pil_img)
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
        response = client.predict(endpoint=model_endpoint, instances=instances, parameters=parameters)

        image_tensors = []
        for prediction in response.predictions:
            base64_image = prediction["bytesBase64Encoded"]
            image_tensors.append(self.base64_to_tensor(base64_image))

        if not image_tensors:
            print("API did not return any images.")
            return (torch.zeros(1, 64, 64, 3, dtype=torch.float32),)

        batch_tensor = torch.cat(image_tensors, 0)
        return (batch_tensor,)

NODE_CLASS_MAPPINGS = {
    "Imagen_Product_Recontext": ImagenProductRecontextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Imagen_Product_Recontext": "Imagen Product Recontext"
}
