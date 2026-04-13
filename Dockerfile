# ── Stage 1: dependency installation ─────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Build-time system deps (compilers + headers for psycopg2, python-magic, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch first to avoid pulling the default CUDA build (~1.5 GB → ~250 MB).
RUN pip install --no-cache-dir --prefix=/install --timeout 120 --retries 5 \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --prefix=/install --timeout 120 --retries 5 -r requirements.txt
# easyocr pulls in opencv-python (non-headless) as a dep; force headless to win.
RUN pip install --no-cache-dir --prefix=/install --timeout 120 --retries 5 --force-reinstall opencv-python-headless


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Runtime system deps — mirrors what was needed on the dev machine for
# Docling, Tesseract OCR, Unstructured, LibreOffice, and friends.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # PostgreSQL client runtime
    libpq5 \
    # OpenCV (headless) runtime libs
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    # OpenMP — required by ONNX Runtime / layout models
    libgomp1 \
    # file-type detection (python-magic / libmagic)
    libmagic1 \
    # PDF rendering / text extraction (Docling, pdf2image, pdfminer)
    poppler-utils \
    # Tesseract OCR engine + language packs
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    # Document format conversion
    pandoc \
    ghostscript \
    unrtf \
    # LibreOffice — needed for DOCX/PPTX/XLSX ingestion via Unstructured
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Pre-download RapidOCR model weights as root so they're baked into the image.
# appuser has no write access to site-packages; downloading at runtime would fail.
RUN python -c "from rapidocr import RapidOCR; RapidOCR()" 2>/dev/null || true

# Non-root user for security
RUN useradd -m -u 1000 appuser \
    && mkdir -p logs storage \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Production: no --reload; workers can be tuned via UVICORN_WORKERS env var
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}"]
