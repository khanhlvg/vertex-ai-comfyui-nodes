import os
import random
import asyncio
from google import genai
from google.genai import types
from .utils import tensor_to_temp_image_file, save_video, save_video_for_preview
import folder_paths
from comfy.comfy_types import IO
from comfy_api.input_impl import VideoFromFile

class VeoNode:
    """
    A custom node that calls the Veo API to generate video from a prompt and an optional image.
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
                "model": (["veo-3.0-fast-generate-preview", "veo-3.0-generate-preview"],),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic shot of a panda eating bamboo."
                }),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "output_file_path": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "duration_seconds": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 16,
                    "step": 1
                }),
                "resolution": (["1080p", "720p"],),
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

    RETURN_TYPES = ("STRING", IO.VIDEO)
    RETURN_NAMES = ("video_file", "video")

    FUNCTION = "generate_video"

    CATEGORY = "Vertex AI"

    def __init__(self):
        self.client = None

    async def generate_video(self, project_id, location, model, prompt, first_frame=None, output_file_path="", duration_seconds=8, resolution="1080p", enhance_prompt=True, generate_audio=True, person_generation="allow_adult", seed=0):
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

        config = types.GenerateVideosConfig(
            number_of_videos=1,
            duration_seconds=duration_seconds,
            resolution=resolution,
            person_generation=person_generation,
            enhance_prompt=enhance_prompt,
            generate_audio=generate_audio,
            seed=seed,
        )

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
            operation = await asyncio.to_thread(
                self.client.models.generate_videos,
                model=model,
                prompt=prompt,
                config=config,
            )

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

        if operation.response:
            video_paths = []
            for i, video in enumerate(operation.result.generated_videos):
                video_bytes = video.video.video_bytes
                if output_file_path:
                    video_path = save_video(video_bytes, f"{output_file_path}_{i}.mp4")
                    video_paths.append(video_path)
                else:
                    video_preview = save_video_for_preview(video_bytes, folder_paths.get_temp_directory())
                    video_paths.append(video_preview["full_path"])

            video_file_path = video_paths[0] if video_paths else None
            
            if not video_file_path:
                return (None, None)

            video_object = VideoFromFile(video_file_path)
            return (video_file_path, video_object)

        return (None, None)

class VeoPromptWriterNode:
    """
    A custom node that takes various video parameters and generates a detailed prompt for the Veo API.
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
        self.client = None

    async def write_prompt(self, project_id, location, subject, action, scene, camera_angle="None", camera_movement="None", lens_effects="None", style="None", temporal_elements="None", sound_effects="None", dialogue=None):
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

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

        gemini_prompt = f"""
        You are an expert video prompt engineer for Google's Veo model. Your task is to construct the most effective and optimal prompt string using the following keywords. Every single keyword MUST be included. Synthesize them into a single, cohesive, and cinematic instruction. Do not add any new core concepts. Output ONLY the final prompt string, without any introduction or explanation. Mandatory Keywords: {",".join(keywords)}
        """
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model="gemini-2.5-flash",
            contents=gemini_prompt,
        )

        return (response.text,)

NODE_CLASS_MAPPINGS = {
    "Veo": VeoNode,
    "Veo_Prompt_Writer": VeoPromptWriterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Veo": "Veo 3",
    "Veo_Prompt_Writer": "Veo Prompt Writer",
}
