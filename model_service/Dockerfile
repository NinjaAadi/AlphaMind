FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p /app/output

# Environment variables (override at runtime with -e)
ENV API_URL=http://host.docker.internal:8000

# Default command - run the pipeline
# Override output path and other args at runtime
CMD ["python", "run_pipeline.py", "--output", "/app/output/training_data.csv", "--force"]
