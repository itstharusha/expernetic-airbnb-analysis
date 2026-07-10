# Use official lightweight Python image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for certain python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit dashboard
CMD ["streamlit", "run", "app/streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
