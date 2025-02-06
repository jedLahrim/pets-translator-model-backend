FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create uploads directory
RUN mkdir -p temp_uploads

# Expose the port the app runs on
EXPOSE 8080

# Use gunicorn as the production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]