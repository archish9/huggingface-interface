FROM python:3.10

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set HF cache dirs to /tmp (writable in HF Spaces)
ENV HF_HOME=/tmp
ENV HF_HUB_CACHE=/tmp

# Upgrade pip
RUN pip install --upgrade pip

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860","--workers", "1"]
