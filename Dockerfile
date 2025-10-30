FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

COPY requirements.txt .
COPY handler.py .
COPY cache_model.py .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

# RUN python cache_model.py 

CMD ["python", "handler.py"]


