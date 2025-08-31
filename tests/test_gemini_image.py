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

from vertex_ai_comfyui_nodes.gemini_image import GeminiImageNode
from vertex_ai_comfyui_nodes.gemini import GeminiCallerNode

class TestGeminiImageNode(unittest.TestCase):
    def setUp(self):
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = "global"
        self.gemini_image_node = GeminiImageNode()
        self.gemini_node = GeminiCallerNode()

    def test_generate_image_and_verify_with_gemini(self):
        # Define the prompt for the image generation
        prompt = "a blue car"

        # Generate the image using the Gemini Image node
        image_tensor, = asyncio.run(self.gemini_image_node.generate_image(
            project_id=self.project_id,
            location=self.location,
            prompt=prompt,
            model_name="gemini-2.5-flash-image-preview",
            seed=0,
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

if __name__ == '__main__':
    unittest.main()
