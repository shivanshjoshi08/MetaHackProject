FROM python:3.11-slim
WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Default: run baseline inference (override via docker run args)
CMD ["python", "baseline_inference.py"]