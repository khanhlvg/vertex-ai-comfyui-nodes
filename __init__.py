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

from .vertex_ai_comfyui_nodes.imagen import NODE_CLASS_MAPPINGS as IMAGEN_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as IMAGEN_DISPLAY_NAME_MAPPINGS
from .vertex_ai_comfyui_nodes.gemini import NODE_CLASS_MAPPINGS as GEMINI_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as GEMINI_DISPLAY_NAME_MAPPINGS
from .vertex_ai_comfyui_nodes.imagen_exp import NODE_CLASS_MAPPINGS as IMAGEN_EXP_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as IMAGEN_EXP_DISPLAY_NAME_MAPPINGS
from .vertex_ai_comfyui_nodes.veo import NODE_CLASS_MAPPINGS as VEO_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VEO_DISPLAY_NAME_MAPPINGS
from .vertex_ai_comfyui_nodes.lyria import NODE_CLASS_MAPPINGS as LYRIA_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LYRIA_DISPLAY_NAME_MAPPINGS
from .vertex_ai_comfyui_nodes.chirp import NODE_CLASS_MAPPINGS as CHIRP_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CHIRP_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {**IMAGEN_CLASS_MAPPINGS, **GEMINI_CLASS_MAPPINGS, **IMAGEN_EXP_CLASS_MAPPINGS, **VEO_CLASS_MAPPINGS, **LYRIA_CLASS_MAPPINGS, **CHIRP_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**IMAGEN_DISPLAY_NAME_MAPPINGS, **GEMINI_DISPLAY_NAME_MAPPINGS, **IMAGEN_EXP_DISPLAY_NAME_MAPPINGS, **VEO_DISPLAY_NAME_MAPPINGS, **LYRIA_DISPLAY_NAME_MAPPINGS, **CHIRP_DISPLAY_NAME_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
