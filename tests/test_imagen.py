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
from vertex_ai_comfyui_nodes.utils import tensor_to_pil

class TestImagenT2INode(unittest.TestCase):
    def setUp(self):
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        self.imagen_node = ImagenT2ICallerNode()
        self.gemini_node = GeminiCallerNode()

    def test_generate_image_and_verify_with_gemini(self):
        # Define the prompt for the image generation
        prompt = "a red car"

        # Generate the image using the Imagen node
        image_tensor, = asyncio.run(self.imagen_node.call_image_api(
            project_id=self.project_id,
            location=self.location,
            prompt=prompt,
            model="imagen-4.0-generate-preview-06-06",
            num_images=1,
            aspect_ratio="1:1",
            seed=0,
            safety_filter_level="BLOCK_ONLY_HIGH",
            person_generation="ALLOW_ALL",
            enhancePrompt=True,
            image_size="1K",
            output_mime_type="image/png"
        ))

        # Verify that the image was generated
        self.assertIsNotNone(image_tensor)

        # Use the Gemini node to verify the image content
        verification_prompt = f"Is there a {prompt} in this image? If so, say '{prompt}'."
        gemini_response, = asyncio.run(self.gemini_node.generate_text(
            project_id=self.project_id,
            region=self.location,
            model_name="gemini-2.5-flash",
            prompt=verification_prompt,
            image1=image_tensor
        ))

        # Assert that the Gemini response contains the original prompt
        self.assertIn(prompt, gemini_response.lower())

    def test_image_size_and_mime_type(self):
        # Generate the image using the Imagen node
        image_tensor, = asyncio.run(self.imagen_node.call_image_api(
            project_id=self.project_id,
            location=self.location,
            prompt="a blue bird",
            model="imagen-4.0-ultra-generate-preview-06-06",
            num_images=1,
            aspect_ratio="1:1",
            seed=0,
            safety_filter_level="BLOCK_ONLY_HIGH",
            person_generation="ALLOW_ALL",
            enhancePrompt=True,
            image_size="2K",
            output_mime_type="image/jpeg"
        ))

        # Verify that the image was generated
        self.assertIsNotNone(image_tensor)
        pil_image = tensor_to_pil(image_tensor)
        self.assertEqual(pil_image.size, (2048, 2048))


    def test_number_of_images(self):
        # Generate the image using the Imagen node
        image_tensor, = asyncio.run(self.imagen_node.call_image_api(
            project_id=self.project_id,
            location=self.location,
            prompt="a dog",
            model="imagen-4.0-generate-preview-06-06",
            num_images=2,
            aspect_ratio="1:1",
            seed=0,
            safety_filter_level="BLOCK_ONLY_HIGH",
            person_generation="ALLOW_ALL",
            enhancePrompt=True,
            image_size="1K",
            output_mime_type="image/png"
        ))

        # Verify that the correct number of images was generated
        self.assertEqual(image_tensor.shape[0], 2)

    def test_safety_filter_handling(self):
        with self.assertRaises(ValueError) as context:
            asyncio.run(self.imagen_node.call_image_api(
                project_id=self.project_id,
                location=self.location,
                prompt="a photo of a child",
                model="imagen-4.0-generate-preview-06-06",
                num_images=1,
                aspect_ratio="1:1",
                seed=0,
                safety_filter_level="BLOCK_ONLY_HIGH",
                person_generation="ALLOW_ADULT",
                enhancePrompt=True,
                image_size="1K",
                output_mime_type="image/png"
            ))
        self.assertIn("No valid images were returned by the API", str(context.exception))


if __name__ == '__main__':
    unittest.main()
