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

import asyncio
import base64
import os
import random

import shortuuid
import torchaudio
from google.cloud import aiplatform, storage

import folder_paths


class LyriaNode:
    """
    A ComfyUI node for generating music using Google's Lyria model.

    This node connects to the Vertex AI API to generate musical pieces from a
    text prompt. It handles the API call, processes the returned audio data,
    and outputs it in a format compatible with other ComfyUI audio nodes.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Lyria node.

        This includes required parameters like the Google Cloud project ID,
        location, and the text prompt, as well as optional settings for GCS
        output, sample count, and a random seed.
        """
        return {
            "required": {
                "project_id": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": os.environ.get("GOOGLE_CLOUD_PROJECT"),
                    },
                ),
                "location": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"),
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "An epic, cinematic soundtrack with a full orchestra, soaring strings, and dramatic percussion.",
                    },
                ),
            },
            "optional": {
                "output_gcs_bucket": ("STRING", {"multiline": False, "default": ""}),
                "sample_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "seed": (
                    "INT",
                    {
                        "default": random.randint(0, 4294967295),
                        "min": 0,
                        "max": 4294967295,
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_music"
    CATEGORY = "Vertex AI"

    def __init__(self):
        """
        Initializes the node by setting the API client to None.
        The client will be created on the first execution.
        """
        self.client = None

    async def generate_music(
        self, project_id, location, prompt, output_gcs_bucket="", sample_count=1, seed=0
    ):
        """
        Generates music by calling the Lyria model in Vertex AI.

        This asynchronous method handles the entire process: initializing the
        client, constructing the API request, calling the prediction endpoint,
        and processing the response. It can save the output to GCS and also
        returns the audio data as a batched tensor for use in ComfyUI.

        Args:
            project_id (str): The Google Cloud project ID.
            location (str): The Google Cloud region.
            prompt (str): The text prompt for music generation.
            output_gcs_bucket (str, optional): The GCS bucket to save the output.
            sample_count (int): The number of music samples to generate.
            seed (int): The random seed for generation.

        Returns:
            tuple: A tuple containing a dictionary with the audio waveform
                   and sample rate, formatted for ComfyUI's AUDIO type.
        """
        # Initialize the AI Platform client with the correct regional endpoint.
        aiplatform.init(project=project_id, location=location)
        api_regional_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_regional_endpoint}

        if self.client is None:
            self.client = aiplatform.gapic.PredictionServiceClient(
                client_options=client_options
            )

        # Construct the request payload.
        instances = [{"prompt": prompt}]
        parameters = {"sampleCount": sample_count, "seed": seed}

        model_endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/lyria-002"

        # Call the prediction endpoint asynchronously.
        response = await asyncio.to_thread(
            self.client.predict,
            endpoint=model_endpoint,
            instances=instances,
            parameters=parameters,
        )

        # Validate the response and extract the audio data.
        if not response.predictions or not response.predictions[0].get(
            "bytesBase64Encoded"
        ):
            error_message = (
                "Lyria API returned an unexpected response (no valid prediction data)."
            )
            if response.predictions and response.predictions[0].get("error"):
                error_detail = response.predictions[0]["error"]
                error_message = f"Lyria API Error: {error_detail.get('message', 'Unknown error from API payload')}"
            raise ValueError(error_message)

        audio_bytes = base64.b64decode(response.predictions[0]["bytesBase64Encoded"])

        # Generate a unique file name for the audio.
        file_name = f"lyria_generation_{shortuuid.uuid()}.wav"

        # If a GCS bucket is specified, upload the audio and download it back
        # to ensure the data being used is the same as the stored version.
        if output_gcs_bucket:
            storage_client = storage.Client(project=project_id)
            bucket = storage_client.bucket(output_gcs_bucket)
            blob = bucket.blob(f"music/{file_name}")
            blob.upload_from_string(audio_bytes, content_type="audio/wav")
            audio_bytes = blob.download_as_bytes()

        # Save the audio to a temporary file.
        temp_dir = folder_paths.get_temp_directory()
        file_name = f"lyria_generation_{shortuuid.uuid()}.wav"
        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(audio_bytes)

        # Load the audio file into a tensor.
        audio_tensor, sample_rate = torchaudio.load(file_path)

        # Add a batch dimension to the tensor to match ComfyUI's expected format.
        audio_tensor = audio_tensor.unsqueeze(0)

        # Return the audio data in the ComfyUI AUDIO format.
        return ({"waveform": audio_tensor, "sample_rate": sample_rate},)


# A dictionary that ComfyUI uses to register the nodes in this file
NODE_CLASS_MAPPINGS = {"Lyria": LyriaNode}

# A dictionary that ComfyUI uses to display the node names in the UI
NODE_DISPLAY_NAME_MAPPINGS = {"Lyria": "Lyria Music Generation"}
