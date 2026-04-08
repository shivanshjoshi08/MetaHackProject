FROM python:3.11-slim
WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Validate OpenEnv compliance at build time
RUN openenv validate openenv.yaml

# Default: run the FastAPI backend for OpenEnv compliance
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]