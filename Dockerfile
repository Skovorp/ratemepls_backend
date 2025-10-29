FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# RunSelector expects the handler to be available
CMD ["python", "handler.py"]

