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
from google import genai
from google.genai import types
from google.genai.types import Part
import io
import asyncio

from .utils import tensor_to_pil

class GeminiCallerNode:
    """
    A ComfyUI node for interacting with the Google Gemini models.

    This node allows users to send text prompts and optional images to the Gemini API
    and receive a generated text response. It supports various Gemini models and
    can be configured with a system instruction.
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
        Defines the input types for the Gemini node, including required fields
        like project ID, region, model, and prompt, as well as optional inputs
        for system instructions and up to three images.
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
                    'gemini-2.5-flash-lite',
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

    async def generate_text(self, project_id, region, model_name, prompt, system_instruction=None, image1=None, image2=None, image3=None):
        """
        Generates text using the Gemini API based on a prompt and optional images.

        This asynchronous method handles the entire process of calling the Gemini API.
        It initializes the client if needed, prepares the request payload by combining
        the text prompt and any provided images, and sends the request to the
        specified Gemini model. The final text response is returned.

        Args:
            project_id (str): The Google Cloud project ID.
            region (str): The Google Cloud region.
            model_name (str): The name of the Gemini model to use.
            prompt (str): The main text prompt.
            system_instruction (str, optional): Instructions for the model's behavior.
            image1 (torch.Tensor, optional): The first input image.
            image2 (torch.Tensor, optional): The second input image.
            image3 (torch.Tensor, optional): The third input image.

        Returns:
            tuple: A tuple containing the generated text string.
        """
        # Initialize the Gemini client if it hasn't been already.
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=region)

        # Create a generation configuration if a system instruction is provided.
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        ) if system_instruction else None

        # Prepare the contents of the request, starting with the text prompt.
        contents = [prompt]
        images = [image1, image2, image3]

        # Process and add any provided images to the request contents.
        for img_tensor in images:
            if img_tensor is not None:
                # Convert the tensor to a PIL Image.
                pil_img = tensor_to_pil(img_tensor)
                # Save the PIL image to an in-memory byte stream.
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                # Add the image bytes to the contents as a Part.
                contents.append(Part.from_bytes(data=img_bytes, mime_type="image/png"))

        # Call the Gemini API asynchronously.
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model_name,
            contents=contents,
            config=config,
        )
        # Return the generated text as a tuple, which is the expected format for ComfyUI.
        return (response.text,)

# A dictionary that ComfyUI uses to register the nodes in this file
NODE_CLASS_MAPPINGS = {
    "Gemini": GeminiCallerNode
}

# A dictionary that ComfyUI uses to display the node names in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini": "Gemini"
}
