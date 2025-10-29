import os
import base64
import io
import numpy as np
import runpod
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

# Initialize model and processor
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m" 
processor = None
model = None
device = None


def initialize_model():
    """Initialize the model and processor on first run."""
    global processor, model, device
    
    print(f"Initializing model: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    model.eval()
    print("Model initialized successfully")


def base64_to_image(base64_string):
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def inference(image):
    """Run inference on the image."""
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract pooling output (the CLS token embedding)
    pooling_output = outputs.pooler_output
    
    # Return average of the pooling vector
    if isinstance(pooling_output, torch.Tensor):
        avg_vector = pooling_output.mean().item()
        return float(avg_vector)
    else:
        # Fallback: compute mean across all dimensions
        avg_vector = pooling_output.mean()
        return float(avg_vector)


def handler(event):
    """
    Handler function for RunPod serverless endpoint.
    Expects event with 'input' containing 'image' (base64 encoded).
    """
    try:
        # Initialize model on first run
        if model is None:
            initialize_model()
        
        # Get base64 image from input
        input_data = event.get('input', {})
        base64_image = input_data.get('image')
        
        if not base64_image:
            return {
                "error": "No image provided. Please provide 'image' in base64 format."
            }
        
        # Decode base64 to image
        image = base64_to_image(base64_image)
        
        # Run inference
        result = inference(image)
        
        return {
            "avg_pooling_vector": result
        }
        
    except Exception as e:
        return {
            "error": str(e)
        }


# Start RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

