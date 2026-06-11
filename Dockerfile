# ==========================================
# STAGE 1: Build Dependencies Environment
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# STAGE 2: Ultra-lean Runtime Deployment Layer
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_PATH=/app/data/autopilot_analytics.db

# Copy installed package environments from builder layer
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Ensure a dedicated data volume folder exists for SQLite file structures
RUN mkdir -p /app/data

# Copy system code components safely into active directory layers
COPY . /app/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
