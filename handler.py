import base64
import io
import runpod
from PIL import Image
import torch
from transformers import AutoModel
import traceback
from torchvision.transforms import v2
import random

MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m" 

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

def make_transform(resize_size):
    to_tensor = v2.ToImage()
    if resize_size is not None:
        resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if resize_size is not None:
        return v2.Compose([to_tensor, resize, to_float, normalize])
    else:
        return v2.Compose([to_tensor, to_float, normalize])


def inference_model(image):
    device = 'cuda'
    my_transform = make_transform(512)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    image = my_transform(image)
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(image).pooler_output
        out = out.mean().item()
        return out

def handler(event):
    """
    Handler function for RunPod serverless endpoint.
    Expects event with 'input' containing 'image' (base64 encoded).
    """
    try:
        input_data = event.get('input', {})
        base64_image = input_data.get('image')
        
        if not base64_image:
            return {
                "error": "No image provided. Please provide 'image' in base64 format."
            }
        
        image = base64_to_image(base64_image)
        result = inference_model(image)
        
        out = round(random.betavariate(3, 3) * 100, 2)
        
        return {
            "avg_pooling_vector": out
        }
        
    except Exception:
        return {
            "error": traceback.format_exc()
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})


