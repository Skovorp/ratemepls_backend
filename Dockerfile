FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

ENV PYTHONUNBUFFERED=1

# RUN python cache_model.py 

CMD ["python", "handler.py"]

