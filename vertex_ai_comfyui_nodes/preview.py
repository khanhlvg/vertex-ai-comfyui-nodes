import os
import folder_paths

class PreviewVideo:
    @classmethod
    def INPUT_TYPES(s):
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
        if not video_path or not os.path.exists(video_path):
            return {"ui": {"images": []}}

        filename = os.path.basename(video_path)
        
        output_dir = folder_paths.get_output_directory()
        input_dir = folder_paths.get_input_directory()
        temp_dir = folder_paths.get_temp_directory()
        
        file_dir = os.path.dirname(video_path)

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
        
        if subfolder == ".":
            subfolder = ""

        video_data = [{"filename": filename, "subfolder": subfolder, "type": type}]
        return {"ui": {"images": video_data, "animated": (True,)}}

NODE_CLASS_MAPPINGS = {
    "PreviewVideo": PreviewVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewVideo": "Preview Video by Path"
}