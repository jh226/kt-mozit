# Python Slim 이미지 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필수 시스템 패키지 설치 (PyTorch 및 기타 패키지 의존성)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    build-essential \
    libopenblas-dev \
    libssl-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# pip, setuptools 최신화
RUN python -m pip install --upgrade pip setuptools wheel

# requirements.txt 복사
COPY requirements.txt /tmp/

# PyTorch와 Transformers를 먼저 설치 (시간 절약 및 캐싱)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ultralytics && \
    pip install --no-cache-dir transformers opencv-python-headless

# 나머지 의존성 설치
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ✅ WebSocket 관련 패키지 추가
RUN pip install "uvicorn[standard]" websockets

# 애플리케이션 코드 복사
COPY . /app

# 포트 노출
EXPOSE 8000

# FastAPI 애플리케이션 실행
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "info"]