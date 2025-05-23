import io

from fastapi import APIRouter, Depends, Request, UploadFile, File, HTTPException
from PIL import Image
import numpy as np

from utils.predictor import predict_skin_condition
from schemas.prediction import ConditionInfo, PredictionResponse
from utils.recommend import get_recommended_products

router = APIRouter()

def get_model(request: Request):
    """
    Dependency that retrieves the preloaded ML model from app state.
    Raises HTTPException if not available.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return model

@router.post(
    "/predict/",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict Skin Condition",
    description=(
        "Upload a skin image to predict the most likely skin condition with confidence. "
        "Recommended products for the detected condition will be included if available."
    ),
)
async def predict(
    file: UploadFile = File(...),
    model = Depends(get_model),
) -> PredictionResponse:
    # Validate content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload an image.")

    try:
        contents = await file.read()
        await file.close()

        # Load and preprocess image
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)

        # Run prediction
        prediction = predict_skin_condition(img_array, model)
        predicted_condition = prediction.get("condition")
        confidence = prediction.get("confidence")

        # Handle no-detection case
        if not predicted_condition:
            raise HTTPException(
                status_code=422,
                detail="Unable to detect a skin condition from the provided image."
            )

        # Build response
        result = {
            "predicted_condition": predicted_condition,
            "confidence": confidence,
            "info": None,
        }

        # Attach recommendations if available
        condition_data = get_recommended_products(predicted_condition)
        if not condition_data:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendation data found for detected condition: {predicted_condition}"
            )

        result["info"] = ConditionInfo(**condition_data)
        return result

    except HTTPException:
        # Propagate HTTPExceptions
        raise
    except Exception:
        # Hide technical details from client
        raise HTTPException(status_code=500, detail="Internal server error")
