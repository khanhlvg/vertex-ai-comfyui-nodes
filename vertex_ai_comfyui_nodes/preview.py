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
import folder_paths

class PreviewVideo:
    """
    A ComfyUI node to display a video preview from a given file path.

    This node is designed to be an output node, meaning it provides a visual
    representation in the ComfyUI interface but doesn't produce an output
    that can be connected to other nodes.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input type for the PreviewVideo node.

        The only required input is the path to the video file.
        """
        return {
            "required": {
                "video_path": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_video"
    OUTPUT_NODE = True
    CATEGORY = "image/video"

    def preview_video(self, video_path):
        """
        Generates the data structure required by ComfyUI to display a video.

        This method takes a video file path, verifies its existence, and then
        determines its location relative to ComfyUI's output, input, or temp
        directories. It then constructs a dictionary that the ComfyUI frontend
        can use to render the video preview.

        Args:
            video_path (str): The absolute path to the video file.

        Returns:
            dict: A dictionary formatted for the ComfyUI frontend to display the video.
        """
        # If the path is invalid or doesn't exist, return an empty UI response.
        if not video_path or not os.path.exists(video_path):
            return {"ui": {"images": []}}

        # Extract the filename from the path.
        filename = os.path.basename(video_path)
        
        # Get the standard ComfyUI directory paths.
        output_dir = folder_paths.get_output_directory()
        input_dir = folder_paths.get_input_directory()
        temp_dir = folder_paths.get_temp_directory()
        
        # Determine the directory of the video file.
        file_dir = os.path.dirname(video_path)

        # Determine the subfolder and type based on the file's location.
        subfolder = ""
        type = "output"

        if file_dir.startswith(output_dir):
            subfolder = os.path.relpath(file_dir, output_dir)
            type = "output"
        elif file_dir.startswith(input_dir):
            subfolder = os.path.relpath(file_dir, input_dir)
            type = "input"
        elif file_dir.startswith(temp_dir):
            subfolder = os.path.relpath(file_dir, temp_dir)
            type = "temp"
        
        # Normalize the subfolder path.
        if subfolder == ".":
            subfolder = ""

        # Create the data structure for the UI.
        video_data = [{"filename": filename, "subfolder": subfolder, "type": type}]
        # Return the UI data, indicating that the content is animated.
        return {"ui": {"images": video_data, "animated": (True,)}}

NODE_CLASS_MAPPINGS = {
    "PreviewVideo": PreviewVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewVideo": "Preview Video by Path"
}