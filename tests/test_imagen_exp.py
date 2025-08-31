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
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vertex_ai_comfyui_nodes.imagen_exp import VirtualTryOnNode, ImagenProductRecontextNode
from vertex_ai_comfyui_nodes.gemini import GeminiCallerNode

def image_to_tensor(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)

class TestVirtualTryOnNode(unittest.TestCase):
    def setUp(self):
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        self.vto_node = VirtualTryOnNode()
        self.gemini_node = GeminiCallerNode()
        self.person_image_path = os.path.join(os.path.dirname(__file__), "res", "person_01.png")
        self.product_image_path = os.path.join(os.path.dirname(__file__), "res", "product_01_dress.png")

    def test_virtual_try_on(self):
        person_tensor = image_to_tensor(self.person_image_path)
        product_tensor = image_to_tensor(self.product_image_path)

        image_tensor, = asyncio.run(self.vto_node.virtual_try_on(
            project_id=self.project_id,
            location=self.location,
            model="virtual-try-on-preview-08-04",
            person_image=person_tensor,
            product_image=product_tensor,
        ))

        self.assertIsNotNone(image_tensor)

        verification_prompt = "Does this image contain a person wearing some clothes? If so, say 'person wearing clothes'."
        gemini_response, = asyncio.run(self.gemini_node.generate_text(
            project_id=self.project_id,
            region=self.location,
            model_name="gemini-2.5-flash",
            prompt=verification_prompt,
            image1=image_tensor
        ))

        self.assertIn("person wearing clothes", gemini_response.lower())

class TestImagenProductRecontextNode(unittest.TestCase):
    def setUp(self):
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        self.recontext_node = ImagenProductRecontextNode()
        self.gemini_node = GeminiCallerNode()
        self.product_image_path = os.path.join(os.path.dirname(__file__), "res", "product_06_handbag.png")

    def test_product_recontext(self):
        product_tensor = image_to_tensor(self.product_image_path)
        prompt = "on a wooden table"

        image_tensor, = asyncio.run(self.recontext_node.generate_contextualized_image(
            project_id=self.project_id,
            location=self.location,
            model="imagen-product-recontext-preview-06-30",
            prompt=prompt,
            image1=product_tensor
        ))

        self.assertIsNotNone(image_tensor)

        verification_prompt = f"Is the object on a wooden table in this image? If so, say 'on a wooden table'."
        gemini_response, = asyncio.run(self.gemini_node.generate_text(
            project_id=self.project_id,
            region=self.location,
            model_name="gemini-2.5-flash",
            prompt=verification_prompt,
            image1=image_tensor
        ))

        self.assertIn("on a wooden table", gemini_response.lower())

if __name__ == '__main__':
    unittest.main()
