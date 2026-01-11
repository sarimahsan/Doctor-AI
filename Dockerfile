# -------------------------
# Stage 1: Builder
# -------------------------
FROM python:3.11-slim AS builder

WORKDIR /Doctor-AI

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Stage 2: Final
# -------------------------
FROM python:3.11-slim

WORKDIR /Doctor-AI

# Copy EVERYTHING python installed
COPY --from=builder /usr/local /usr/local

# Copy backend
COPY backend ./backend

EXPOSE 8000

ENV TRANSFORMERS_CACHE=/Doctor-AI/models

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
