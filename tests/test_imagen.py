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

import unittest
import os
import sys
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vertex_ai_comfyui_nodes.imagen import ImagenT2ICallerNode
from vertex_ai_comfyui_nodes.gemini import GeminiCallerNode
from vertex_ai_comfyui_nodes.utils import tensor_to_pil, pil_to_base64

class TestImagenT2INode(unittest.TestCase):
    def test_generate_image_and_verify_with_gemini(self):
        # Get project ID and location from environment variables
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

        # Instantiate the Imagen and Gemini nodes
        imagen_node = ImagenT2ICallerNode()
        gemini_node = GeminiCallerNode()

        # Define the prompt for the image generation
        prompt = "a red car"

        # Generate the image using the Imagen node
        image_tensor = asyncio.run(imagen_node.call_image_api(
            project_id=project_id,
            location=location,
            prompt=prompt,
            model="imagen-4.0-generate-preview-06-06",
            num_images=1,
            aspect_ratio="1:1",
            seed=0,
            safety_filter_level="BLOCK_ONLY_HIGH",
            person_generation="ALLOW_ALL",
            enhancePrompt=True
        ))

        # Verify that the image was generated
        self.assertIsNotNone(image_tensor)

        # Use the Gemini node to verify the image content
        verification_prompt = f"Is there a {prompt} in this image? If so, say '{prompt}'."
        gemini_response = asyncio.run(gemini_node.generate_text(
            project_id=project_id,
            region=location,
            model_name="gemini-2.0-flash",
            prompt=verification_prompt,
            image1=image_tensor[0]
        ))

        # Assert that the Gemini response contains the original prompt
        self.assertIn(prompt, gemini_response[0].lower())

if __name__ == '__main__':
    unittest.main()