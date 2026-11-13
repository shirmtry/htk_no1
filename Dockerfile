FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake curl libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# Torch CPU-only: tránh tải ~5GB wheel CUDA không dùng được trên máy CPU
RUN pip install --upgrade pip && pip install --index-url https://download.pytorch.org/whl/cpu torch
RUN CMAKE_ARGS="-DLLAMA_NATIVE=ON" pip install -r requirements.txt
COPY app.py run_eval.py ./
COPY data/ ./data/
COPY models/ ./models/
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
