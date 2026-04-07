# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /👋_Introduction

# Avoid Python buffering issues
ENV PYTHONUNBUFFERED=1

# Install system dependencies (optional but often needed)
RUN apt-get update && apt-get install -y build-essential curl \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "👋_Introduction.py", "--server.port=8501", "--server.address=0.0.0.0"]