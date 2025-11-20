# Dockerfile for ML Model Training
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy UV binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY model/ ./model/
COPY analyse/ ./analyse/
COPY test_pipeline.py ./

# Create directories for model outputs
RUN mkdir -p /app/model/trained_models /app/model/processed_data /app/analyse

# Run the training pipeline
CMD ["uv", "run", "python", "test_pipeline.py"]
