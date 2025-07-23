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
import torch
import numpy as np
from PIL import Image
import base64
import io
import tempfile

def tensor_to_pil(tensor):
    """
    Converts a torch.Tensor to a PIL Image.

    This function handles the conversion of a tensor, which is a common data
    format in machine learning frameworks, to a PIL Image object. It correctly
    handles the tensor's data type and value range.

    Args:
        tensor (torch.Tensor): The input tensor, expected to be in a format
                               compatible with image representation.

    Returns:
        PIL.Image or None: The converted PIL Image, or None if the input is None.
    """
    if tensor is None:
        return None
    # Squeeze the tensor to remove single-dimensional entries from the shape.
    image_np = tensor.squeeze().cpu().numpy()
    # Normalize and convert to uint8 if the data is in the [0, 1] range.
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    return Image.fromarray(image_np)

def pil_to_base64(pil_image):
    """
    Converts a PIL Image to a base64 encoded string.

    This is useful for serializing image data for transmission over a network,
    for example, in an API request.

    Args:
        pil_image (PIL.Image): The input PIL Image.

    Returns:
        str or None: The base64 encoded string, or None if the input is None.
    """
    if pil_image is None:
        return None
    # Create an in-memory binary stream.
    buffered = io.BytesIO()
    # Save the image to the stream in PNG format.
    pil_image.save(buffered, format="PNG")
    # Encode the binary data to a base64 string.
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_tensor(base64_image):
    """
    Converts a base64 encoded image string back to a torch.Tensor.

    This function reverses the process of `pil_to_base64` and `tensor_to_pil`,
    allowing the system to work with image data received from an API.

    Args:
        base64_image (str): The base64 encoded image string.

    Returns:
        torch.Tensor: The resulting image tensor.
    """
    # Decode the base64 string to bytes.
    image_data = base64.b64decode(base64_image)
    # Open the image data as a PIL Image.
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
    # Convert the PIL Image to a NumPy array and normalize to [0, 1].
    image_array = np.array(pil_image).astype(np.float32) / 255.0
    # Convert the NumPy array to a torch.Tensor and add a batch dimension.
    return torch.from_numpy(image_array)[None,]

def tensor_to_temp_image_file(tensor):
    """
    Saves a tensor as a temporary image file.

    This is useful when an API or library requires a file path to an image
    instead of image data in memory.

    Args:
        tensor (torch.Tensor): The input image tensor.

    Returns:
        str: The path to the temporary image file.
    """
    pil_image = tensor_to_pil(tensor)
    # Create a named temporary file with a .png suffix.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        pil_image.save(temp_file.name)
        return temp_file.name

def save_video(video_bytes, file_path):
    """
    Saves video data to a file.

    Args:
        video_bytes (bytes): The video data in bytes.
        file_path (str): The path where the video will be saved.

    Returns:
        str: The path to the saved video file.
    """
    with open(file_path, "wb") as out_file:
        out_file.write(video_bytes)
    return file_path

def save_video_for_preview(video_bytes, output_dir, file_path=None):
    """
    Saves video data and prepares it for preview in ComfyUI.

    This function can either save the video to a specified path or create a
    temporary file. It returns a dictionary formatted for the ComfyUI previewer.

    Args:
        video_bytes (bytes): The video data.
        output_dir (str): The directory to save the video in if no path is given.
        file_path (str, optional): The specific file path to save the video to.

    Returns:
        dict: A dictionary containing information for the ComfyUI previewer.
    """
    if file_path:
        with open(file_path, "wb") as out_file:
            out_file.write(video_bytes)
        return {
            "filename": os.path.basename(file_path),
            "subfolder": "",
            "type": "output"
        }
    else:
        # Create a temporary file in the specified output directory.
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=output_dir) as temp_file:
            temp_file.write(video_bytes)
            return {
                "filename": os.path.basename(temp_file.name),
                "subfolder": "",
                "type": "output",
                "full_path": temp_file.name
            }

def save_audio_for_preview(audio_bytes, output_dir, file_name):
    """
    Saves audio data and prepares it for preview in ComfyUI.

    Args:
        audio_bytes (bytes): The audio data.
        output_dir (str): The directory to save the audio in.
        file_name (str): The name of the audio file.

    Returns:
        dict: A dictionary containing information for the ComfyUI previewer.
    """
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "wb") as out_file:
        out_file.write(audio_bytes)
    return {
        "filename": file_name,
        "subfolder": "",
        "type": "output",
        "full_path": file_path
    }
