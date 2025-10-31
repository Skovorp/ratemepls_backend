"""
Test script for local development.
This simulates what RunPod will send to the handler.
"""

import base64
from handler import handler
import os

# Load a test image (replace with your own image path)
def load_test_image():
    # Create a simple test image
    from PIL import Image
    import io
    
    # Create a red square
    img = Image.new('RGB', (224, 224), color='red')
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return img_base64

if __name__ == "__main__":
    print("Testing handler locally...")
    
    # Create test event
    test_event = {
        "input": {
            "image": load_test_image()
        }
    }
    
    # Run handler
    result = handler(test_event)
    
    print("\nResult:")
    print(result)

