# Dockerfile for Web Scraper
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Chromium
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome binary location for undetected-chromedriver
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Copy UV binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Copy scraper code
COPY scrapper/ ./scrapper/

# Create output directory
RUN mkdir -p /app/data

# Run the complete scraper pipeline (URLs + details)
CMD ["uv", "run", "python", "scrapper/run_scraper.py"]
