import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.routes.predict import router

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "flower_model.h5")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")


def _download_model():
    if os.path.exists(MODEL_PATH):
        return
    if not HF_MODEL_REPO:
        raise RuntimeError(
            "Model not found at models/flower_model.keras and HF_MODEL_REPO env var is not set."
        )
    from huggingface_hub import hf_hub_download
    print(f"Downloading model from Hugging Face Hub: {HF_MODEL_REPO} ...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="flower_model.h5",
        local_dir=os.path.dirname(MODEL_PATH),
    )
    print("Model downloaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _download_model()
    from src.predict import load_model
    app.state.model = load_model()
    yield


app = FastAPI(title="Flower Classifier", lifespan=lifespan)

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve React build in production (mounted last so API routes take priority)
_frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_frontend_dist):
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="static")
