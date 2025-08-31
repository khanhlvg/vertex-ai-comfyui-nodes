import unittest
import os
import sys
import asyncio
import torch
import cv2
import numpy as np
from PIL import Image
from unittest.mock import MagicMock
import tempfile

# Mock ComfyUI specific modules
sys.modules['folder_paths'] = MagicMock()
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.comfy_types'] = MagicMock()
sys.modules['comfy_api'] = MagicMock()
sys.modules['comfy_api.input_impl'] = MagicMock()

import folder_paths  # noqa: E402

class MockVideoFromFile:
    def __init__(self, video_path):
        self._video_path = video_path

    def get_stream_source(self):
        return self._video_path

sys.modules['comfy_api.input_impl'].VideoFromFile = MockVideoFromFile

# Set up a mock for folder_paths
folder_paths.get_temp_directory.return_value = tempfile.gettempdir()

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vertex_ai_comfyui_nodes.veo import Veo2Node, Veo3Node, VeoPromptWriterNode, Veo2Extend  # noqa: E402
from vertex_ai_comfyui_nodes.gemini import GeminiCallerNode  # noqa: E402

def pil_to_tensor(pil_image):
    """Converts a PIL image to a torch tensor."""
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np).unsqueeze(0)

class TestVeoNode(unittest.TestCase):
    def setUp(self):
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            self.fail("GOOGLE_CLOUD_PROJECT environment variable is not set.")

        self.gcs_bucket_name = os.environ.get("GCS_BUCKET_NAME")
        if not self.gcs_bucket_name:
            self.fail("GCS_BUCKET_NAME environment variable is not set.")

        self.location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        self.gcs_prefix = "comfyui-node-unittest"

        self.veo2_node = Veo2Node()
        self.veo3_node = Veo3Node()
        self.veo2_extend_node = Veo2Extend()
        self.prompt_writer_node = VeoPromptWriterNode()
        self.gemini_node = GeminiCallerNode()

        self.i2v_test_image_path = os.path.join(os.path.dirname(__file__), "res", "veo_i2v_test.jpg")
        self.extend_test_video_path = os.path.join(os.path.dirname(__file__), "res", "veo2_extend_test.mp4")

    def _video_to_tensor(self, video_path):
        """Extracts the middle frame of a video and converts it to a tensor."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video file.")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_index = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Error: Could not read frame from video.")

        # Convert frame from BGR to RGB and then to a PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        return pil_to_tensor(pil_image)

    def test_veo_prompt_writer(self):
        subject = "a robot"
        action = "dancing"
        scene = "futuristic city"

        prompt, = asyncio.run(self.prompt_writer_node.write_prompt(
            project_id=self.project_id,
            location=self.location,
            subject=subject,
            action=action,
            scene=scene
        ))

        self.assertIsInstance(prompt, str)
        for keyword in subject.split():
            self.assertIn(keyword, prompt.lower())
        self.assertIn("danc", prompt.lower())
        self.assertIn("city", prompt.lower())

    def test_generate_video_and_verify_with_gemini(self):
        prompt = "a majestic eagle soaring through the mountains"

        video_object, = asyncio.run(self.veo3_node.generate_video(
            project_id=self.project_id,
            location=self.location,
            model="veo-3.0-generate-preview",
            prompt=prompt,
        ))

        self.assertIsNotNone(video_object)

        video_path = video_object.get_stream_source()
        frame_tensor = self._video_to_tensor(video_path)

        verification_prompt = "Is there an eagle in this image? Answer with only 'yes' or 'no'."
        gemini_response, = asyncio.run(self.gemini_node.generate_text(
            project_id=self.project_id,
            region=self.location,
            model_name="gemini-2.5-flash",
            prompt=verification_prompt,
            image1=frame_tensor
        ))

        self.assertEqual("yes", gemini_response.lower().strip())

    def test_generate_video_from_image(self):
        image = Image.open(self.i2v_test_image_path)
        image_tensor = pil_to_tensor(image)
        prompt = "a lion running in a desert"

        # Verify input image
        verification_prompt_input = "Is there a lion in this image? Answer with only 'yes' or 'no'."
        gemini_response_input, = asyncio.run(self.gemini_node.generate_text(
            project_id=self.project_id,
            region=self.location,
            model_name="gemini-2.5-flash",
            prompt=verification_prompt_input,
            image1=image_tensor
        ))
        self.assertEqual("yes", gemini_response_input.lower().strip())

        video_object, = asyncio.run(self.veo3_node.generate_video(
            project_id=self.project_id,
            location=self.location,
            model="veo-3.0-generate-preview",
            prompt=prompt,
            first_frame=image_tensor
        ))

        self.assertIsNotNone(video_object)

        video_path = video_object.get_stream_source()
        frame_tensor = self._video_to_tensor(video_path)

        verification_prompt_output = "Is there a lion in this image? Answer with only 'yes' or 'no'."
        gemini_response_output, = asyncio.run(self.gemini_node.generate_text(
            project_id=self.project_id,
            region=self.location,
            model_name="gemini-2.5-flash",
            prompt=verification_prompt_output,
            image1=frame_tensor
        ))

        self.assertEqual("yes", gemini_response_output.lower().strip())


    def test_veo2_generate_video(self):
        prompt = "a beautiful sunset over the ocean"

        video_object, = asyncio.run(self.veo2_node.generate_video(
            project_id=self.project_id,
            location=self.location,
            model="veo-2.0-generate-001",
            prompt=prompt,
        ))

        self.assertIsNotNone(video_object)

    def test_veo2_generate_video_from_image(self):
        image = Image.open(self.i2v_test_image_path)
        image_tensor = pil_to_tensor(image)
        prompt = "a lion running in a desert"

        # Verify input image
        verification_prompt_input = "Is there a lion in this image? Answer with only 'yes' or 'no'."
        gemini_response_input, = asyncio.run(self.gemini_node.generate_text(
            project_id=self.project_id,
            region=self.location,
            model_name="gemini-2.5-flash",
            prompt=verification_prompt_input,
            image1=image_tensor
        ))
        self.assertEqual("yes", gemini_response_input.lower().strip())

        video_object, = asyncio.run(self.veo2_node.generate_video(
            project_id=self.project_id,
            location=self.location,
            model="veo-2.0-generate-001",
            prompt=prompt,
            first_frame=image_tensor
        ))

        self.assertIsNotNone(video_object)

        video_path = video_object.get_stream_source()
        frame_tensor = self._video_to_tensor(video_path)

        verification_prompt_output = "Is there a lion in this image? Answer with only 'yes' or 'no'."
        gemini_response_output, = asyncio.run(self.gemini_node.generate_text(
            project_id=self.project_id,
            region=self.location,
            model_name="gemini-2.5-flash",
            prompt=verification_prompt_output,
            image1=frame_tensor
        ))

        self.assertEqual("yes", gemini_response_output.lower().strip())

    def test_veo2_extend_video(self):
        video_object = MockVideoFromFile(self.extend_test_video_path)
        prompt = "the lion continues to run in the desert"
        temp_gcs_prefix_for_input_video = f"gs://{self.gcs_bucket_name}/{self.gcs_prefix}"
        output_gcs_uri = f"gs://{self.gcs_bucket_name}/{self.gcs_prefix}/"

        extended_video_object, = asyncio.run(self.veo2_extend_node.extend_video(
            project_id=self.project_id,
            location=self.location,
            model="veo-2.0-generate-001",
            prompt=prompt,
            video=video_object,
            temp_gcs_prefix_for_input_video=temp_gcs_prefix_for_input_video,
            output_gcs_uri=output_gcs_uri
        ))

        self.assertIsNotNone(extended_video_object)

if __name__ == '__main__':
    unittest.main()
