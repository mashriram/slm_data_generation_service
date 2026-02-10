FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Required by huggingface/tokenizers
ENV TOKENIZERS_PARALLELISM=false

# Install system dependencies if any (e.g., for certain ML libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Copy only the requirements file to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app /app/app

# Port the service will run on
EXPOSE 8001

# Command to run the application using Uvicorn
# Use --workers 1 for this CPU/GPU-bound task in a container, and scale by running more containers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]    
