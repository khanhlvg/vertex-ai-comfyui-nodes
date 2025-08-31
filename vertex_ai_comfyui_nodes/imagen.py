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
import numpy as np
from PIL import Image
import os
import random
import asyncio

from google import genai
from google.genai import types

from .utils import pil_to_base64, base64_to_tensor, tensor_to_temp_image_file

class ImagenT2ICallerNode:
    """
    A ComfyUI node for generating images from text prompts using the Google Imagen API.

    This node provides a user-friendly interface within ComfyUI to access Imagen's
    text-to-image capabilities. It supports various models, aspect ratios, and advanced
    options like safety filtering and prompt enhancement.
    """
    # Define the input types for your node
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Imagen node.

        This method specifies all the parameters that users can configure in the
        ComfyUI interface, such as the prompt, number of images, model selection,
        and safety settings. Default values are provided for convenience.
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
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A majestic lion in the savannah, 4k, high detail"
                }),
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"],),
                "model": ([
                    "imagen-4.0-generate-001",
                    "imagen-4.0-ultra-generate-001",
                    "imagen-4.0-fast-generate-001",
                    ],),
                "image_size": (["1K", "2K"],),
                "seed": ("INT", {
                    "default": random.randint(0, 4294967295),
                    "min": 0,
                    "max": 4294967295
                }),
                "safety_filter_level": (["BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE", "BLOCK_NONE"],),
                "person_generation": (["ALLOW_ALL", "ALLOW_ADULT", "DONT_ALLOW"],),
                "enhancePrompt": ("BOOLEAN", {"default": True}),
                "output_mime_type": (["image/png", "image/jpeg"],),
            },
        }

    # Define the return types of your node
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    # Define the function that will be executed when the node is called
    FUNCTION = "call_image_api"

    # Define the category for your node
    CATEGORY = "Vertex AI"

    def __init__(self):
        """
        Initializes the node by setting the client to None.
        The client will be created on the first execution.
        """
        self.client = None

    async def call_image_api(self, project_id, location, prompt, model, num_images, aspect_ratio, seed, safety_filter_level, person_generation, enhancePrompt, image_size, output_mime_type):
        """
        This function is called when the node is executed.
        It sends the parameters to the specified API URL and returns the generated images.

        This asynchronous method handles the entire process of calling the Imagen API.
        It initializes the client, constructs the request with all the specified
        parameters, and then processes the response to return the generated images
        as a batch of tensors.

        Args:
            project_id (str): The Google Cloud project ID.
            location (str): The Google Cloud region.
            prompt (str): The text prompt for image generation.
            model (str): The Imagen model to use.
            num_images (int): The number of images to generate.
            aspect_ratio (str): The desired aspect ratio of the images.
            seed (int): The random seed for generation.
            safety_filter_level (str): The safety filtering level.
            person_generation (str): The setting for person generation.
            enhancePrompt (bool): Whether to enhance the prompt.
            image_size (str): The resolution of the image, 1K or 2K.
            output_mime_type (str): The output mime type of the image.

        Returns:
            tuple: A tuple containing a batch of generated images as a torch.Tensor.
        """
        # Initialize the Imagen client if it hasn't been already.
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

        config_args = {
            "aspect_ratio": aspect_ratio,
            "number_of_images": num_images,
            "seed": seed,
            "safety_filter_level": safety_filter_level,
            "person_generation": person_generation,
            "enhance_prompt": enhancePrompt,
            "add_watermark": False,
            "output_mime_type": output_mime_type,
            "include_rai_reason": True,
        }

        is_2K_supported = model.startswith("imagen-4.0-generate") or model.startswith("imagen-4.0-ultra-generate")

        if is_2K_supported:
            config_args["image_size"] = image_size
        elif image_size != "1K":
            raise ValueError(f"Image size '{image_size}' is not supported for model '{model}'. It is only supported for Imagen 4 and Imagen 4 Ultra.")

        # Asynchronously call the Imagen API to generate images.
        api_response = await asyncio.to_thread(
            self.client.models.generate_images,
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(**config_args),
        )

        # Process the generated images and convert them to tensors.
        image_tensors = []
        for image in api_response.generated_images:
            if not image.image.image_bytes:
                print(f"No image returned by the API. Your request was likely blocked by the safety filters. Reason: {image.rai_filtered_reason}")
                continue
            try:
                # Convert the returned image to a PIL Image in RGBA format.
                pil_image = image.image._pil_image.convert("RGBA")
                # Convert the PIL image to a base64 string and then to a tensor.
                image_tensors.append(base64_to_tensor(pil_to_base64(pil_image)))
            except (ValueError, AttributeError):
                # This can happen if the image data is corrupted.
                print("Skipping an image that could not be decoded.")
                continue

        # If no images were successfully processed, raise an error.
        if not image_tensors:
            raise ValueError("No valid images were returned by the API. Your request was likely blocked by the safety filters.")

        # Stack all individual image tensors into a single batch tensor.
        batch_tensor = torch.cat(image_tensors, 0)

        # Return the batch of images as a tuple.
        return (batch_tensor,)

class ImagenMaskEditingNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "project_id": ("STRING", {"multiline": False, "default": os.environ.get("GOOGLE_CLOUD_PROJECT")}),
                "location": ("STRING", {"multiline": False, "default": os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
                "model": (["imagen-3.0-capability-001", "imagen-3.0-capability-002"],),
                "edit_mode": (["EDIT_MODE_INPAINT_REMOVAL", "EDIT_MODE_INPAINT_INSERTION", "EDIT_MODE_BGSWAP", "EDIT_MODE_OUTPAINT"],),
            },
            "optional": {
                "mask": ("MASK",),
                "computed_mask": ("IMAGEN_COMPUTED_MASK",),
                "mask_dilation": ("FLOAT", {"default": -1, "min": -1, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": random.randint(0, 4294967295), "min": 0, "max": 4294967295}),
                "base_steps": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
                "guidance_scale": ("INT", {"default": -1, "min": -1, "max": 500, "step": 1}),
                "number_of_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "safety_filter_level": (["BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE", "BLOCK_NONE"],),
                "person_generation": (["ALLOW_ALL", "ALLOW_ADULT", "DONT_ALLOW"],),
                "output_mime_type": (["image/png", "image/jpeg"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "Vertex AI"

    def __init__(self):
        self.client = None

    async def edit_image(self, project_id, location, prompt, image, model, edit_mode, mask_dilation, seed, base_steps, guidance_scale, number_of_images, safety_filter_level, person_generation, output_mime_type="image/png", mask=None, computed_mask=None):
        if self.client is None:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)

        if mask is None and computed_mask is None:
            raise ValueError("Either a mask or a computed_mask must be provided.")
        if mask is not None and computed_mask is not None:
            raise ValueError("Only one of mask or computed_mask can be provided.")

        image_path = tensor_to_temp_image_file(image)
        raw_ref_image = types.RawReferenceImage(
            reference_image=types.Image.from_file(location=image_path), reference_id=0
        )

        if mask is not None:
            mask_tensor = mask.squeeze(0)
            mask_pil = Image.fromarray((mask_tensor.numpy() * 255).astype(np.uint8), 'L')
            mask_pil = mask_pil.point(lambda p: 255 if p > 127 else 0)
            mask_path = tensor_to_temp_image_file(torch.from_numpy(np.array(mask_pil)).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).float() / 255.0)

            mask_config_args = {"mask_mode": "MASK_MODE_USER_PROVIDED"}
            if mask_dilation != -1:
                mask_config_args["mask_dilation"] = mask_dilation

            mask_ref_image = types.MaskReferenceImage(
                reference_id=1,
                reference_image=types.Image.from_file(location=mask_path),
                config=types.MaskReferenceConfig(**mask_config_args),
            )
        else:
            mask_ref_image = computed_mask
            if mask_dilation != -1:
                mask_ref_image.config.mask_dilation = mask_dilation

        config_args = {
            "edit_mode": edit_mode,
            "number_of_images": number_of_images,
            "seed": seed,
            "safety_filter_level": safety_filter_level,
            "person_generation": person_generation,
            "output_mime_type": output_mime_type,
        }
        if base_steps != -1:
            config_args["base_steps"] = base_steps
        if guidance_scale != -1:
            config_args["guidance_scale"] = guidance_scale

        edited_image = await asyncio.to_thread(
            self.client.models.edit_image,
            model=model,
            prompt=prompt,
            reference_images=[raw_ref_image, mask_ref_image],
            config=types.EditImageConfig(**config_args),
        )

        os.remove(image_path)
        if mask is not None:
            os.remove(mask_path)

        image_tensors = []
        for image in edited_image.generated_images:
            if not image.image.image_bytes:
                print(f"No image returned by the API. Your request was likely blocked by the safety filters. Reason: {image.rai_filtered_reason}")
                continue
            try:
                pil_image = image.image._pil_image.convert("RGBA")
                image_tensors.append(base64_to_tensor(pil_to_base64(pil_image)))
            except (ValueError, AttributeError):
                print("Skipping an image that could not be decoded.")
                continue

        if not image_tensors:
            raise ValueError("No valid images were returned by the API. Your request was likely blocked by the safety filters.")

        batch_tensor = torch.cat(image_tensors, 0)
        return (batch_tensor,)

class ImagenComputedMaskConfigNode:
    SEMANTIC_CLASSES = {
        0: "backpack", 50: "carrot", 100: "sidewalk_pavement", 150: "skis",
        1: "umbrella", 51: "hot_dog", 101: "runway", 151: "snowboard",
        2: "bag", 52: "pizza", 102: "terrain", 152: "sports_ball",
        3: "tie", 53: "donut", 103: "book", 153: "kite",
        4: "suitcase", 54: "cake", 104: "box", 154: "baseball_bat",
        5: "case", 55: "fruit_other", 105: "clock", 155: "baseball_glove",
        6: "bird", 56: "food_other", 106: "vase", 156: "skateboard",
        7: "cat", 57: "chair_other", 107: "scissors", 157: "surfboard",
        8: "dog", 58: "armchair", 108: "plaything_other", 158: "tennis_racket",
        9: "horse", 59: "swivel_chair", 109: "teddy_bear", 159: "net",
        10: "sheep", 60: "stool", 110: "hair_dryer", 160: "base",
        11: "cow", 61: "seat", 111: "toothbrush", 161: "sculpture",
        12: "elephant", 62: "couch", 112: "painting", 162: "column",
        13: "bear", 63: "trash_can", 113: "poster", 163: "fountain",
        14: "zebra", 64: "potted_plant", 114: "bulletin_board", 164: "awning",
        15: "giraffe", 65: "nightstand", 115: "bottle", 165: "apparel",
        16: "animal_other", 66: "bed", 116: "cup", 166: "banner",
        17: "microwave", 67: "table", 117: "wine_glass", 167: "flag",
        18: "radiator", 68: "pool_table", 118: "knife", 168: "blanket",
        19: "oven", 69: "barrel", 119: "fork", 169: "curtain_other",
        20: "toaster", 70: "desk", 120: "spoon", 170: "shower_curtain",
        21: "storage_tank", 71: "ottoman", 121: "bowl", 171: "pillow",
        22: "conveyor_belt", 72: "wardrobe", 122: "tray", 172: "towel",
        23: "sink", 73: "crib", 123: "range_hood", 173: "rug_floormat",
        24: "refrigerator", 74: "basket", 124: "plate", 174: "vegetation",
        25: "washer_dryer", 75: "chest_of_drawers", 125: "person", 175: "bicycle",
        26: "fan", 76: "bookshelf", 126: "rider_other", 176: "car",
        27: "dishwasher", 77: "counter_other", 127: "bicyclist", 177: "autorickshaw",
        28: "toilet", 78: "bathroom_counter", 128: "motorcyclist", 178: "motorcycle",
        29: "bathtub", 79: "kitchen_island", 129: "paper", 179: "airplane",
        30: "shower", 80: "door", 130: "streetlight", 180: "bus",
        31: "tunnel", 81: "light_other", 131: "road_barrier", 181: "train",
        32: "bridge", 82: "lamp", 132: "mailbox", 182: "truck",
        33: "pier_wharf", 83: "sconce", 133: "cctv_camera", 183: "trailer",
        34: "tent", 84: "chandelier", 134: "junction_box", 184: "boat_ship",
        35: "building", 85: "mirror", 135: "traffic_sign", 185: "slow_wheeled_object",
        36: "ceiling", 86: "whiteboard", 136: "traffic_light", 186: "river_lake",
        37: "laptop", 87: "shelf", 137: "fire_hydrant", 187: "sea",
        38: "keyboard", 88: "stairs", 138: "parking_meter", 188: "water_other",
        39: "mouse", 89: "escalator", 139: "bench", 189: "swimming_pool",
        40: "remote", 90: "cabinet", 140: "bike_rack", 190: "waterfall",
        41: "cell phone", 91: "fireplace", 141: "billboard", 191: "wall",
        42: "television", 92: "stove", 142: "sky", 192: "window",
        43: "floor", 93: "arcade_machine", 143: "pole", 193: "window_blind",
        44: "stage", 94: "gravel", 144: "fence",
        45: "banana", 95: "platform", 145: "railing_banister",
        46: "apple", 96: "playingfield", 146: "guard_rail",
        47: "sandwich", 97: "railroad", 147: "mountain_hill",
        48: "orange", 98: "road", 148: "rock",
        49: "broccoli", 99: "snow", 149: "frisbee",
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_mode": (["MASK_MODE_BACKGROUND", "MASK_MODE_FOREGROUND", "MASK_MODE_SEMANTIC"],),
            },
            "optional": {
                "semantic_class": (list(s.SEMANTIC_CLASSES.values()),),
            }
        }

    RETURN_TYPES = ("IMAGEN_COMPUTED_MASK",)
    FUNCTION = "create_config"
    CATEGORY = "Vertex AI"

    def create_config(self, mask_mode, semantic_class=None):
        if mask_mode == "MASK_MODE_SEMANTIC" and semantic_class is None:
            raise ValueError("Semantic class must be provided for MASK_MODE_SEMANTIC.")

        config_args = {"mask_mode": mask_mode}
        if mask_mode == "MASK_MODE_SEMANTIC":
            config_args["segmentation_classes"] = [self.get_class_id(semantic_class)]

        return (types.MaskReferenceImage(
            reference_id=1,
            reference_image=None,
            config=types.MaskReferenceConfig(**config_args),
        ),)

    def get_class_id(self, class_name):
        for class_id, name in self.SEMANTIC_CLASSES.items():
            if name == class_name:
                return class_id
        return -1

# A dictionary that ComfyUI uses to register the nodes in this file
NODE_CLASS_MAPPINGS = {
    "Imagen_T2I": ImagenT2ICallerNode,
    "ImagenMaskEditing": ImagenMaskEditingNode,
    "ImagenComputedMaskConfig": ImagenComputedMaskConfigNode,
}

# A dictionary that ComfyUI uses to display the node names in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "Imagen_T2I": "Imagen Text-to-Image",
    "ImagenMaskEditing": "Imagen Mask Editing",
    "ImagenComputedMaskConfig": "Imagen Compute Mask",
}
