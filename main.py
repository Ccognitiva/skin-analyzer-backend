from config import CLASS_NAMES
import os, logging, numpy as np, asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
from PIL import Image
import gradio as gr

from api.v1 import router as v1_router
from models.model_loader import load_skin_condition_model

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Downloading model from Hugging Face Hub…")
        model_path = hf_hub_download(
            repo_id="yasyn14/skin-analyzer",
            filename="model-v1.keras",
            cache_dir="/app/.cache"
        )

        logger.info("Loading model…")
        model = await asyncio.to_thread(load_skin_condition_model, model_path)

        # warm-up
        dummy = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        await asyncio.to_thread(model.predict, dummy)

        app.state.model = model
        logger.info("Model ready ✅")
        yield

    except Exception as e:
        logger.exception("Failed during startup:")
        raise RuntimeError("Failed to load skin-condition model") from e

    finally:
        logger.info("Shutting down: releasing resources")
        if hasattr(app.state, "model"):
            del app.state.model

# === FastAPI Setup ===
app = FastAPI(
    lifespan=lifespan,
    title="Skin Condition Classifier API",
    description="Upload skin images to detect skin conditions using AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz", tags=["Health"])
async def health_check():
    return {"status": "ok"}

app.include_router(v1_router)

# === Gradio UI Setup ===
def predict_skin_condition(image: Image.Image):
    if image is None:
        return "No image provided"

    model = app.state.model

    # Preprocess image
    img = image.resize((224, 224)).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    input_data = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Predict
    prediction = model.predict(input_data)[0]

    # Map prediction to label (replace with your actual class labels)
    top_idx = np.argmax(prediction)
    confidence = float(prediction[top_idx])
    label = CLASS_NAMES[top_idx]

    return f"{label} ({confidence:.2%} confidence)"

gradio_interface = gr.Interface(
    fn=predict_skin_condition,
    inputs=gr.Image(type="pil", label="Upload a skin image"),
    outputs=gr.Text(label="Prediction"),
    title="Skin Analyzer",
    description="Upload a photo of skin to detect conditions like acne, eczema, dryness, etc."
)

# Mount Gradio on root
app = gr.mount_gradio_app(app, gradio_interface, path="/")