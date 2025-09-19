# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Update system and install Python 3.10 and dependencies
RUN apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python3.10 to be accessible as python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create app directory
RUN mkdir -p /app

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files to /app
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p /app/preprocessing /app/predict /app/models

# Set Python path to include the app directory
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port 8000 (FastAPI with Uvicorn default)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3", "app.py"]