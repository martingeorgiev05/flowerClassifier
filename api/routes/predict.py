import io
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

router = APIRouter()

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_DIMENSION = 8000

class PredictionResponse(BaseModel):
    flower: str
    confidence: float

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: Request, file: UploadFile = File(...)):
    # 1. Extension check
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(422, "Unsupported file type. Allowed: JPEG, PNG, WEBP")

    # 2. MIME type check
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(422, "Unsupported file type. Allowed: JPEG, PNG, WEBP")

    # 3. Size check
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, "File exceeds 5 MB limit")

    # 4. Open image and check dimensions
    try:
        img = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(422, "Could not read image")

    if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
        raise HTTPException(422, f"Image dimensions too large (max {MAX_DIMENSION}×{MAX_DIMENSION})")

    # 5. Run inference
    from src.predict import predict_from_pil
    flower, confidence = predict_from_pil(img, request.app.state.model)
    return PredictionResponse(flower=flower, confidence=confidence)
