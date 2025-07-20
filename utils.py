import torch
import numpy as np
from PIL import Image
import base64
import io

def tensor_to_pil(tensor):
    if tensor is None:
        return None
    image_np = tensor.squeeze().cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    return Image.fromarray(image_np)

def pil_to_base64(pil_image):
    if pil_image is None:
        return None
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_tensor(base64_image):
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
    image_array = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array)[None,]
