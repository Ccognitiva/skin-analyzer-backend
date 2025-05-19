# syntax=docker/dockerfile:1            # enables COPY --chown and other BuildKit features
FROM python:3.12-slim

# ─────────────────────────────────────────────
# 1. Create a non‑root user early (UID 1000 is standard)
# ─────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# ─────────────────────────────────────────────
# 2. Environment variables
#    • Cache directories live inside the user’s home
#    • Add ~/.local/bin to PATH for anything installed with --user
# ─────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/transformers \
    HF_HUB_CACHE=/home/appuser/.cache/huggingface/hub \
    PORT=8000 \
    PATH=/home/appuser/.local/bin:$PATH

# ─────────────────────────────────────────────
# 3. Set working directory inside the user’s home
# ─────────────────────────────────────────────
WORKDIR /home/appuser/app

# ─────────────────────────────────────────────
# 4. Install Python dependencies *as root* (keeps image small)
# ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# ─────────────────────────────────────────────
# 5. Download model during build
#    (still running as root, then we fix ownership)
# ─────────────────────────────────────────────
RUN python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="yasyn14/skin-analyzer",
    filename="model-v1.keras",
    cache_dir="/home/appuser/.cache"
)
PY

# ─────────────────────────────────────────────
# 6. Copy the rest of the application and give it to the user
#    • Requires BuildKit (Docker 20.10+) for --chown
# ─────────────────────────────────────────────
COPY --chown=appuser:appuser . .

# Make sure anything created earlier is user‑writable
RUN chown -R appuser:appuser /home/appuser

# ─────────────────────────────────────────────
# 7. Switch to the non‑root user for runtime
# ─────────────────────────────────────────────
USER appuser

# ─────────────────────────────────────────────
# 8. Expose port & launch
# ─────────────────────────────────────────────
EXPOSE ${PORT}
CMD ["sh", "-c", "fastapi run main.py --host 0.0.0.0 --port $PORT"]
