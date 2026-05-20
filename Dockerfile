# Pingi-AI Dockerfile

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -r -s /usr/sbin/nologin appuser \
    && mkdir -p /app/tmp \
    && chown -R appuser:appuser /app

USER appuser

ENV PORT=8001
ENV TEMP_DIR=/app/tmp

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
