# RateMePls Backend - RunPod Serverless

A serverless API endpoint for image analysis using Facebook's DINOv2 model.

## Features

- Single POST endpoint `rate_this_image`
- Accepts base64 encoded images
- Uses Facebook DINOv2-base model
- Returns average of pooling vector
- Runs on RTX 4000 with PyTorch

## Setup

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the handler:
```bash
python handler.py
```

### RunPod Deployment

**Option 1: Using Custom Dockerfile**
1. Create a new serverless endpoint on RunPod
2. Upload this repository as a zip file
3. Set handler path to: `handler`
4. Select GPU: RTX 4000
5. Use the provided Dockerfile (automatically detected)
6. Configure environment variables if needed

**Option 2: Using Pre-built Image**
1. Create a new serverless endpoint on RunPod
2. Upload this repository as a zip file
3. Set handler path to: `handler`
4. Select GPU: RTX 4000
5. Set container image: `runpod/pytorch:2.1.0-cuda11.8.0-devel`
6. Configure environment variables if needed

## API Usage

### Endpoint
`POST /rate_this_image`

### Request
```json
{
  "input": {
    "image": "base64_encoded_image_string"
  }
}
```

### Response
```json
{
  "avg_pooling_vector": 0.123456
}
```

### Error Response
```json
{
  "error": "Error message"
}
```

## Model

- Model: `facebook/dinov2-base`
- Task: Image feature extraction
- Output: Average of pooling vector from CLS token

## Dependencies

- `runpod==1.3.0` - RunPod serverless SDK
- `torch>=2.0.0` - PyTorch for model inference
- `transformers>=4.35.0` - Hugging Face transformers
- `pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Numerical operations
- `einops>=0.7.0` - Einstein notation for tensors

## Notes

- Model is initialized on first request (cold start)
- Automatic GPU detection and CUDA usage
- Images are automatically converted to RGB format
- Model is set to evaluation mode for inference

