FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libmagic1 \
    build-essential \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY schemas/ ./schemas/
COPY proto/ ./proto/
COPY app.py .
COPY models/ ./models/
COPY utils/ ./utils/
COPY appcontext.py .
COPY dependencies/ ./dependencies/
COPY db/ ./db/
COPY tests/ ./tests/
COPY data/ ./data/
COPY .env . 

# Create directory for uploads

# Download InsightFace models
# need to cache this

# Expose the port the app will run on

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "4"]