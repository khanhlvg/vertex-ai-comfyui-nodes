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
import asyncio
import shortuuid
import torchaudio
from google.cloud import texttospeech
import folder_paths

# A mapping from user-friendly language names to BCP-47 language codes.
LANGUAGE_MAP = {
    "Arabic (Generic)": "ar-XA",
    "Bengali (India)": "bn-IN",
    "Dutch (Belgium)": "nl-BE",
    "Dutch (Netherlands)": "nl-NL",
    "English (Australia)": "en-AU",
    "English (India)": "en-IN",
    "English (United Kingdom)": "en-GB",
    "English (United States)": "en-US",
    "French (Canada)": "fr-CA",
    "French (France)": "fr-FR",
    "German (Germany)": "de-DE",
    "Gujarati (India)": "gu-IN",
    "Hindi (India)": "hi-IN",
    "Indonesian (Indonesia)": "id-ID",
    "Italian (Italy)": "it-IT",
    "Japanese (Japan)": "ja-JP",
    "Kannada (India)": "kn-IN",
    "Korean (South Korea)": "ko-KR",
    "Malayalam (India)": "ml-IN",
    "Mandarin Chinese (China)": "cmn-CN",
    "Marathi (India)": "mr-IN",
    "Polish (Poland)": "pl-PL",
    "Portuguese (Brazil)": "pt-BR",
    "Russian (Russia)": "ru-RU",
    "Spanish (Spain)": "es-ES",
    "Spanish (United States)": "es-US",
    "Swahili (Kenya)": "sw-KE",
    "Tamil (India)": "ta-IN",
    "Telugu (India)": "te-IN",
    "Thai (Thailand)": "th-TH",
    "Turkish (Turkey)": "tr-TR",
    "Ukrainian (Ukraine)": "uk-UA",
    "Urdu (India)": "ur-IN",
    "Vietnamese (Vietnam)": "vi-VN",
}

# A list of the available Chirp3 HD voice names.
VOICE_NAMES = [
    "Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Aoede", "Autonoe",
    "Callirrhoe", "Charon", "Despina", "Enceladus", "Erinome", "Fenrir",
    "Gacrux", "Iapetus", "Kore", "Laomedeia", "Leda", "Orus", "Pulcherrima",
    "Puck", "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar", "Sulafat",
    "Umbriel", "Vindemiatrix", "Zephyr", "Zubenelgenubi",
]

class ChirpNode:
    """
    A ComfyUI node for synthesizing speech using Google's Chirp3 HD voices.

    This node provides a user-friendly interface for Google's Text-to-Speech
    API, allowing users to select a language and voice from dropdown menus
    and generate high-quality audio from text.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the Chirp node.

        This includes dropdown menus for selecting the language and voice,
        a text area for the input text, and fields for the Google Cloud
        project ID and location.
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
                "language": (list(LANGUAGE_MAP.keys()),),
                "voice_name": (VOICE_NAMES,),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello there this is Vertex AI Chirp 3 HD from a ComfyUI node!"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "synthesize_speech"
    CATEGORY = "Vertex AI"

    def __init__(self):
        """
        Initializes the node by setting the API client to None.
        The client will be created on the first execution.
        """
        self.client = None

    async def synthesize_speech(self, project_id, location, language, voice_name, text):
        """
        Synthesizes speech from text using the selected language and voice.

        This asynchronous method constructs the full voice name from the user's
        selections, calls the Text-to-Speech API, and returns the generated
        audio in a format compatible with ComfyUI.

        Args:
            project_id (str): The Google Cloud project ID.
            location (str): The Google Cloud region.
            language (str): The user-friendly language name.
            voice_name (str): The selected voice name.
            text (str): The text to be synthesized.

        Returns:
            tuple: A tuple containing a dictionary with the audio waveform
                   and sample rate, formatted for ComfyUI's AUDIO type.
        """
        if self.client is None:
            self.client = texttospeech.TextToSpeechClient()

        # Construct the full voice name from the language and voice name.
        language_code = LANGUAGE_MAP[language]
        full_voice_name = f"{language_code}-Chirp3-HD-{voice_name}"

        input_text = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=full_voice_name,
        )

        # Request linear16 audio, which is uncompressed and suitable for processing.
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        # Call the API asynchronously.
        response = await asyncio.to_thread(
            self.client.synthesize_speech,
            input=input_text,
            voice=voice,
            audio_config=audio_config,
        )

        # Save the audio to a temporary file.
        temp_dir = folder_paths.get_temp_directory()
        file_name = f"chirp_generation_{shortuuid.uuid()}.wav"
        file_path = os.path.join(temp_dir, file_name)

        with open(file_path, "wb") as out:
            out.write(response.audio_content)

        # Load the audio file into a tensor.
        audio_tensor, sample_rate = torchaudio.load(file_path)

        # Add a batch dimension to the tensor.
        audio_tensor = audio_tensor.unsqueeze(0)

        # Return the audio data in the ComfyUI AUDIO format.
        return ({"waveform": audio_tensor, "sample_rate": sample_rate},)

# A dictionary that ComfyUI uses to register the nodes in this file
NODE_CLASS_MAPPINGS = {
    "Chirp": ChirpNode
}

# A dictionary that ComfyUI uses to display the node names in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "Chirp": "Chirp 3: HD Voices"
}
