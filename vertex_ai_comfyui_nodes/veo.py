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
import random
import asyncio
import uuid
from google import genai
from google.genai import types
from google.cloud import storage
from .utils import tensor_to_temp_image_file, save_video_for_preview
import folder_paths
from comfy.comfy_types import IO
from comfy_api.input_impl import VideoFromFile

class Veo3Node:
    """
    A ComfyUI node for generating video from a prompt and an optional image using the Veo 3 API.

    This node supports the latest features of Veo 3, including higher resolution,
    longer duration, and audio generation. It can take a text prompt and an
    optional first frame to guide the video generation process.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Veo 3 node.

        This includes required parameters like project ID, location, model, and
        prompt, as well as optional settings for the first frame, output GCS URI,
        duration, resolution, and other generation controls.
        """
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
                "model": (["veo-3.0-fast-generate-preview", "veo-3.0-generate-preview"],),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic shot of a panda eating bamboo."
                }),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "output_gcs_uri": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "duration_seconds": ("INT", {
                    "default": 8,
                    "min": 8,
                    "max": 8,
                    "step": 1
                }),
                "resolution": (["1080p", "720p"],),
                "compression_quality": (["OPTIMIZED", "LOSSLESS"],),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "person_generation": (["allow_adult", "dont_allow", "allow_all"],),
                "seed": ("INT", {
                    "default": random.randint(0, 4294967295),
                    "min": 0,
                    "max": 4294967295
                }),
            }
        }

    RETURN_TYPES = (IO.VIDEO,)
    RETURN_NAMES = ("video",)

    FUNCTION = "generate_video"

    CATEGORY = "Vertex AI"

    def __init__(self):
        """
        Initializes the node by setting the client to None.
        The client will be created on the first execution.
        """
        self.client = None

    async def generate_video(self, project_id, location, model, prompt, first_frame=None, output_gcs_uri=None, duration_seconds=8, resolution="1080p", compression_quality="OPTIMIZED", enhance_prompt=True, generate_audio=True, person_generation="allow_adult", seed=0):
        """
        Generates a video using the Veo 3 API.

        This asynchronous method handles the entire video generation process. It
        initializes the client, prepares the configuration, and calls the Veo API.
        It supports both image-to-video and text-to-video generation and handles
        the asynchronous nature of the video generation operation.

        Args:
            project_id (str): The Google Cloud project ID.
            location (str): The Google Cloud region.
            model (str): The Veo 3 model to use.
            prompt (str): The text prompt for the video.
            first_frame (torch.Tensor, optional): The first frame of the video.
            output_gcs_uri (str, optional): GCS URI to save the output video.
            duration_seconds (int): The duration of the video in seconds.
            resolution (str): The resolution of the video.
            compression_quality (str): The compression quality of the video.
            enhance_prompt (bool): Whether to enhance the prompt.
            generate_audio (bool): Whether to generate audio for the video.
            person_generation (str): The setting for person generation.
            seed (int): The random seed for generation.

        Returns:
            tuple: A tuple containing the generated video as a VideoFromFile object.
        """
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

        # Configure the video generation parameters.
        config = {
            "number_of_videos": 1,
            "duration_seconds": duration_seconds,
            "resolution": resolution,
            "person_generation": person_generation,
            "enhance_prompt": enhance_prompt,
            "generate_audio": generate_audio,
            "seed": seed,
        }

        if compression_quality == "LOSSLESS":
            config["compression_quality"] = types.VideoCompressionQuality.LOSSLESS
        else:
            config["compression_quality"] = types.VideoCompressionQuality.OPTIMIZED

        if output_gcs_uri:
            config["output_gcs_uri"] = output_gcs_uri

        config = types.GenerateVideosConfig(**config)

        # Handle image-to-video generation if a first frame is provided.
        image_path = None
        if first_frame is not None:
            image_path = tensor_to_temp_image_file(first_frame)
            image_file = types.Image.from_file(location=image_path)
            operation = await asyncio.to_thread(
                self.client.models.generate_videos,
                model=model,
                prompt=prompt,
                image=image_file,
                config=config,
            )
        else:
            # Handle text-to-video generation.
            operation = await asyncio.to_thread(
                self.client.models.generate_videos,
                model=model,
                prompt=prompt,
                config=config,
            )

        # Poll the operation until it is complete.
        while not operation.done:
            await asyncio.sleep(8)
            operation = await asyncio.to_thread(
                self.client.operations.get,
                operation
            )

        if image_path:
            os.remove(image_path)

        if operation.error:
            raise ValueError(operation.error["message"])

        # Process the response and return the video.
        if operation.response:
            if output_gcs_uri:
                video_uri = operation.result.generated_videos[0].video.uri

                # Download the video from GCS.
                storage_client = storage.Client(project=project_id)
                bucket_name, blob_name = video_uri.replace("gs://", "").split("/", 1)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                video_bytes = blob.download_as_bytes()

                # Save the video for preview and return it as a VideoFromFile object.
                video_preview = save_video_for_preview(video_bytes, folder_paths.get_temp_directory())
                video_object = VideoFromFile(video_preview["full_path"])
                return (video_object,)
            else:
                # Handle videos returned directly as bytes.
                video_paths = []
                for i, video in enumerate(operation.result.generated_videos):
                    video_bytes = video.video.video_bytes
                    video_preview = save_video_for_preview(video_bytes, folder_paths.get_temp_directory())
                    video_paths.append(video_preview["full_path"])

                video_file_path = video_paths[0] if video_paths else None

                if not video_file_path:
                    return (None,)

                video_object = VideoFromFile(video_file_path)
                return (video_object,)

        return (None,)

class Veo2Node:
    """
    A ComfyUI node for generating video using the Veo 2 API.

    This node is tailored for the Veo 2 model, providing access to its specific
    features and limitations. It allows for text-to-video and image-to-video
    generation with options for aspect ratio and other parameters.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Veo 2 node.

        This includes required parameters like project ID, location, model, prompt,
        and an output GCS URI. Optional inputs are available for the first frame,
        duration, aspect ratio, and other settings.
        """
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
                "model": (["veo-2.0-generate-001"],),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic shot of a panda eating bamboo."
                }),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "output_gcs_uri": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "duration_seconds": ("INT", {
                    "default": 8,
                    "min": 5,
                    "max": 8,
                    "step": 1
                }),
                "aspect_ratio": (["16:9", "9:16"],),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
                "person_generation": (["allow_adult", "dont_allow"],),
                "seed": ("INT", {
                    "default": random.randint(0, 4294967295),
                    "min": 0,
                    "max": 4294967295
                }),
            }
        }

    RETURN_TYPES = (IO.VIDEO,)
    RETURN_NAMES = ("video",)

    FUNCTION = "generate_video"

    CATEGORY = "Vertex AI"

    def __init__(self):
        """
        Initializes the node by setting the client to None.
        The client will be created on the first execution.
        """
        self.client = None

    async def generate_video(self, project_id, location, model, prompt, first_frame=None, last_frame=None, output_gcs_uri=None, duration_seconds=8, aspect_ratio="16:9", enhance_prompt=True, person_generation="allow_adult", seed=0):
        """
        Generates a video using the Veo 2 API.

        This asynchronous method is similar to the Veo 3 version but is tailored
        for the Veo 2 model. It handles the API call, polling for completion,
        and processing the response to return the generated video.

        Args:
            project_id (str): The Google Cloud project ID.
            location (str): The Google Cloud region.
            model (str): The Veo 2 model to use.
            prompt (str): The text prompt for the video.
            first_frame (torch.Tensor, optional): The first frame of the video.
            last_frame (torch.Tensor, optional): The last frame of the video.
            output_gcs_uri (str, optional): GCS URI to save the output video.
            duration_seconds (int): The duration of the video in seconds.
            aspect_ratio (str): The aspect ratio of the video.
            enhance_prompt (bool): Whether to enhance the prompt.
            person_generation (str): The setting for person generation.
            seed (int): The random seed for generation.

        Returns:
            tuple: A tuple containing the generated video as a VideoFromFile object.
        """
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

        # Configure the video generation parameters for Veo 2.
        config = {
            "number_of_videos": 1,
            "duration_seconds": duration_seconds,
            "aspect_ratio": aspect_ratio,
            "person_generation": person_generation,
            "enhance_prompt": enhance_prompt,
            "seed": seed,
        }

        if output_gcs_uri:
            config["output_gcs_uri"] = output_gcs_uri

        last_frame_path = None
        if last_frame is not None:
            last_frame_path = tensor_to_temp_image_file(last_frame)
            config["last_frame"] = types.Image.from_file(location=last_frame_path)

        config = types.GenerateVideosConfig(**config)

        # Handle image-to-video generation if a first frame is provided.
        image_path = None
        if first_frame is not None:
            image_path = tensor_to_temp_image_file(first_frame)
            image_file = types.Image.from_file(location=image_path)
            operation = await asyncio.to_thread(
                self.client.models.generate_videos,
                model=model,
                prompt=prompt,
                image=image_file,
                config=config,
            )
        else:
            # Handle text-to-video generation.
            operation = await asyncio.to_thread(
                self.client.models.generate_videos,
                model=model,
                prompt=prompt,
                config=config,
            )

        # Poll the operation until it is complete.
        while not operation.done:
            await asyncio.sleep(15)
            operation = await asyncio.to_thread(
                self.client.operations.get,
                operation
            )

        if image_path:
            os.remove(image_path)
        if last_frame_path:
            os.remove(last_frame_path)

        if operation.error:
            raise ValueError(operation.error["message"])

        # Process the response and return the video.
        if operation.response:
            if output_gcs_uri:
                video_uri = operation.result.generated_videos[0].video.uri

                # Download the video from GCS.
                storage_client = storage.Client(project=project_id)
                bucket_name, blob_name = video_uri.replace("gs://", "").split("/", 1)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                video_bytes = blob.download_as_bytes()

                # Save the video for preview and return it.
                video_preview = save_video_for_preview(video_bytes, folder_paths.get_temp_directory())
                video_object = VideoFromFile(video_preview["full_path"])
                return (video_object,)
            else:
                # Handle videos returned directly as bytes.
                video_bytes = operation.result.generated_videos[0].video.video_bytes
                video_preview = save_video_for_preview(video_bytes, folder_paths.get_temp_directory())
                video_object = VideoFromFile(video_preview["full_path"])
                return (video_object,)

        return (None,)


class Veo2Extend(Veo2Node):
    """
    A ComfyUI node for extending a video using the Veo 2 API.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Veo 2 Extend node.
        """
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
                "model": (["veo-2.0-generate-001"],),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "a butterfly flies in and lands on the flower"
                }),
                "video": (IO.VIDEO,),
                "temp_gcs_prefix_for_input_video": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "output_gcs_uri": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
            },
            "optional": {
                "duration_seconds": ("INT", {
                    "default": 7,
                    "min": 4,
                    "max": 7,
                    "step": 1
                }),
                "aspect_ratio": (["16:9", "9:16"],),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
                "person_generation": (["allow_adult", "dont_allow"],),
                "seed": ("INT", {
                    "default": random.randint(0, 4294967295),
                    "min": 0,
                    "max": 4294967295
                }),
            }
        }

    FUNCTION = "extend_video"

    async def extend_video(self, project_id, location, model, prompt, video, temp_gcs_prefix_for_input_video, output_gcs_uri, duration_seconds=7, aspect_ratio="16:9", enhance_prompt=True, person_generation="allow_adult", seed=0):
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

        storage_client = storage.Client(project=project_id)
        bucket_name, prefix = temp_gcs_prefix_for_input_video.replace("gs://", "").split("/", 1)

        # Ensure the prefix ends with a slash to denote a folder
        if not prefix.endswith("/"):
            prefix += "/"

        file_name = f"{prefix}{uuid.uuid4()}.mp4"

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(video.get_stream_source())
        video_uri = f"gs://{bucket_name}/{file_name}"

        config = {
            "number_of_videos": 1,
            "duration_seconds": duration_seconds,
            "aspect_ratio": aspect_ratio,
            "person_generation": person_generation,
            "enhance_prompt": enhance_prompt,
            "seed": seed,
        }

        if output_gcs_uri:
            config["output_gcs_uri"] = output_gcs_uri

        config = types.GenerateVideosConfig(**config)

        operation = await asyncio.to_thread(
            self.client.models.generate_videos,
            model=model,
            prompt=prompt,
            video=types.Video(uri=video_uri),
            config=config,
        )

        while not operation.done:
            await asyncio.sleep(15)
            operation = await asyncio.to_thread(
                self.client.operations.get,
                operation
            )

        if operation.error:
            raise ValueError(operation.error["message"])

        if operation.response:
            if output_gcs_uri:
                video_uri = operation.result.generated_videos[0].video.uri

                storage_client = storage.Client(project=project_id)
                bucket_name, blob_name = video_uri.replace("gs://", "").split("/", 1)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                video_bytes = blob.download_as_bytes()

                video_preview = save_video_for_preview(video_bytes, folder_paths.get_temp_directory())
                video_object = VideoFromFile(video_preview["full_path"])
                return (video_object,)
            else:
                video_bytes = operation.result.generated_videos[0].video.video_bytes
                video_preview = save_video_for_preview(video_bytes, folder_paths.get_temp_directory())
                video_object = VideoFromFile(video_preview["full_path"])
                return (video_object,)

        return (None,)

class VeoPromptWriterNode:
    """
    A ComfyUI node that uses Gemini to generate a detailed video prompt for Veo.

    This node takes various video parameters (subject, action, scene, etc.) and
    uses the Gemini API to synthesize them into a single, effective prompt for
    the Veo model. This helps users create more cinematic and detailed videos.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Veo Prompt Writer node.

        This includes required inputs for the subject, action, and scene, as well
        as a wide range of optional parameters for camera work, style, and sound.
        """
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
                "subject": ("STRING", {
                    "multiline": True,
                    "default": "a detective"
                }),
                "action": ("STRING", {
                    "multiline": True,
                    "default": "interrogating a rubber duck"
                }),
                "scene": ("STRING", {
                    "multiline": True,
                    "default": "in a dark interview room"
                }),
            },
            "optional": {
                "camera_angle": (["None", "Eye-Level Shot", "Low-Angle Shot", "High-Angle Shot", "Bird's-Eye View", "Top-Down Shot", "Worm's-Eye View", "Dutch Angle", "Canted Angle", "Close-Up", "Extreme Close-Up", "Medium Shot", "Full Shot", "Long Shot", "Wide Shot", "Establishing Shot", "Over-the-Shoulder Shot", "Point-of-View (POV) Shot"],),
                "camera_movement": (["None", "Static Shot (or fixed)", "Pan (left)", "Pan (right)", "Tilt (up)", "Tilt (down)", "Dolly (In)", "Dolly (Out)", "Zoom (In)", "Zoom (Out)", "Truck (Left)", "Truck (Right)", "Pedestal (Up)", "Pedestal (Down)", "Crane Shot", "Aerial Shot", "Drone Shot", "Handheld", "Shaky Cam", "Whip Pan", "Arc Shot"],),
                "lens_effects": (["None", "Wide-Angle Lens (e.g., 24mm)", "Telephoto Lens (e.g., 85mm)", "Shallow Depth of Field", "Bokeh", "Deep Depth of Field", "Lens Flare", "Rack Focus", "Fisheye Lens Effect", "Vertigo Effect (Dolly Zoom)"],),
                "style": (["None", "Photorealistic", "Cinematic", "Vintage", "Japanese anime style", "Claymation style", "Stop-motion animation", "In the style of Van Gogh", "Surrealist painting", "Monochromatic black and white", "Vibrant and saturated", "Film noir style", "High-key lighting", "Low-key lighting", "Golden hour glow", "Volumetric lighting", "Backlighting to create a silhouette"],),
                "temporal_elements": (["None", "Slow-motion", "Fast-paced action", "Time-lapse", "Hyperlapse", "Pulsating light", "Rhythmic movement"],),
                "sound_effects": (["None", "Sound of a phone ringing", "Water splashing", "Soft house sounds", "Ticking clock", "City traffic and sirens", "Waves crashing", "Quiet office hum"],),
                "dialogue": ("STRING", {
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    FUNCTION = "write_prompt"

    CATEGORY = "Vertex AI"

    def __init__(self):
        """
        Initializes the node by setting the client to None.
        The client will be created on the first execution.
        """
        self.client = None

    async def write_prompt(self, project_id, location, subject, action, scene, camera_angle="None", camera_movement="None", lens_effects="None", style="None", temporal_elements="None", sound_effects="None", dialogue=None):
        """
        Writes a detailed video prompt using the Gemini API.

        This asynchronous method takes a set of keywords and uses Gemini to
        generate a cohesive and cinematic prompt for the Veo model. It combines
        all the provided elements into a single, effective instruction.

        Args:
            project_id (str): The Google Cloud project ID.
            location (str): The Google Cloud region.
            subject (str): The main subject of the video.
            action (str): The action taking place.
            scene (str): The setting of the video.
            camera_angle (str, optional): The camera angle.
            camera_movement (str, optional): The camera movement.
            lens_effects (str, optional): Any lens effects.
            style (str, optional): The artistic style of the video.
            temporal_elements (str, optional): Any temporal effects.
            sound_effects (str, optional): Any sound effects.
            dialogue (str, optional): Any dialogue.

        Returns:
            tuple: A tuple containing the generated prompt string.
        """
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

        # Collect all the keywords for the prompt.
        keywords = [subject, action, scene]
        optional_keywords = [
            camera_angle,
            camera_movement,
            lens_effects,
            style,
            temporal_elements,
            sound_effects,
        ]
        for keyword in optional_keywords:
            if keyword != "None":
                keywords.append(keyword)
        if dialogue:
            keywords.append(dialogue)

        # Construct the prompt for Gemini to generate the Veo prompt.
        gemini_prompt = f'''
        You are an expert video prompt engineer for Google's Veo model. Your task is to construct the most effective and optimal prompt string using the following keywords. Every single keyword MUST be included. Synthesize them into a single, cohesive, and cinematic instruction. Do not add any new core concepts. Output ONLY the final prompt string, without any introduction or explanation. Mandatory Keywords: {",".join(keywords)}
        '''
        # Call the Gemini API to generate the prompt.
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model="gemini-2.5-flash",
            contents=gemini_prompt,
        )

        # Return the generated prompt.
        return (response.text,)

NODE_CLASS_MAPPINGS = {
    "Veo3": Veo3Node,
    "Veo2": Veo2Node,
    "Veo2Extend": Veo2Extend,
    "Veo_Prompt_Writer": VeoPromptWriterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Veo3": "Veo 3 Video Generation",
    "Veo2": "Veo 2 Video Generation",
    "Veo2Extend": "Veo 2 Video Extend",
    "Veo_Prompt_Writer": "Veo Prompt Writer",
}
