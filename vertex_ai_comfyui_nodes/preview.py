import os
import folder_paths
from comfy.comfy_types import IO

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
                "video": (IO.VIDEO,),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_video"
    OUTPUT_NODE = True
    CATEGORY = "image/video"

    def preview_video(self, video):
        """
        Generates the data structure required by ComfyUI to display a video.

        This method takes a video file path, verifies its existence, and then
        determines its location relative to ComfyUI's output, input, or temp
        directories. It then constructs a dictionary that the ComfyUI frontend
        can use to render the video preview.

        Args:
            video (VideoFromFile): The video object to preview.

        Returns:
            dict: A dictionary formatted for the ComfyUI frontend to display the video.
        """
        video_path = video.get_stream_source()
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
    "PreviewVideo": PreviewVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewVideo": "Preview Video",
}
