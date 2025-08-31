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

import torch
from PIL import Image
import os
import random
import asyncio
import io

from google import genai
from google.genai import types
from google.genai.types import Part

from .utils import tensor_to_pil, pil_to_base64, base64_to_tensor

class GeminiImageNode:
    """
    A ComfyUI node for generating images using the Google Gemini 2.5 Flash Image model.

    This node allows users to generate images from a text prompt and up to six
    input images. It supports configuring the number of images to generate,
    the seed for reproducibility, and other parameters.
    """

    def __init__(self):
        """
        Initializes the node by setting the client to None.
        The client will be created on the first execution.
        """
        self.client = None

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Gemini Image node.
        """
        return {
            "required": {
                "project_id": ("STRING", {
                    "multiline": False,
                    "default": os.environ.get("GOOGLE_CLOUD_PROJECT")
                }),
                "location": ("STRING", {
                    "multiline": False,
                    "default": "global"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape painting."
                }),
                "model_name": (["gemini-2.5-flash-image-preview"],),
                "seed": ("INT", {
                    "default": random.randint(0, 2147483647),
                    "min": 0,
                    "max": 2147483647
                }),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_image"
    CATEGORY = "Vertex AI"

    async def generate_image(self, project_id, location, prompt, model_name, seed,
                             image1=None, image2=None, image3=None, image4=None,
                             image5=None, image6=None):
        """
        Generates images using the Gemini API based on a prompt and optional images.
        """
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

        contents = []
        images = [image1, image2, image3, image4, image5, image6]

        for img_tensor in images:
            if img_tensor is not None:
                pil_img = tensor_to_pil(img_tensor)
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                contents.append(Part.from_bytes(data=img_bytes, mime_type="image/png"))

        contents.append(prompt)

        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            candidate_count=1,
            seed=seed,
        )

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model_name,
            contents=contents,
            config=config,
        )

        image_tensors = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data:
                    try:
                        pil_image = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                        image_tensors.append(base64_to_tensor(pil_to_base64(pil_image)))
                    except (ValueError, AttributeError):
                        print("Skipping an image that could not be decoded.")
                        continue

        if not image_tensors:
            raise ValueError("No valid images were returned by the API. Your request was likely blocked by the safety filters.")

        batch_tensor = torch.cat(image_tensors, 0)
        return (batch_tensor,)

NODE_CLASS_MAPPINGS = {
    "GeminiImage": GeminiImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImage": "Gemini Image"
}
