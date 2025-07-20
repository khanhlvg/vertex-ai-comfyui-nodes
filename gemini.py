import os
from google import genai
from google.genai import types

class Gemini:
    def __init__(self):
        self.client = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "project_id": ("STRING", {
                    "multiline": False,
                    "default": os.environ.get("GCP_PROJECT_ID")
                }),
                "region": ("STRING", {
                    "multiline": False,
                    "default": os.environ.get("GCP_REGION")
                }),
                "model_name": ([
                    'gemini-2.5-flash',
                    'gemini-2.5-pro',
                    'gemini-2.0-flash',
                    'gemini-2.0-flash-lite',
                    'gemini-2.5-flash-lite-preview-06-17',
                ],),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A photo of a cat.",
                }),
            },
            "optional": {
                "system_instruction": ("STRING", {
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"

    CATEGORY = "Vertex AI"

    def generate_text(self, project_id, region, model_name, prompt, system_instruction=None):
        if self.client is None:
            self.client = genai.Client(project=project_id, location=region)
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        ) if system_instruction else None

        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        return (response.text,)

NODE_CLASS_MAPPINGS = {
    "Gemini": Gemini
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini": "Gemini"
}