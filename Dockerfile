# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for numpy, sklearn, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

RUN pip install --no-cache-dir -e .
# Expose port for Cloud Run
EXPOSE 8080

# Run FastAPI with uvicorn
CMD ["uvicorn", "src.model_inferencing.api:app", "--host", "0.0.0.0", "--port", "8080"]
