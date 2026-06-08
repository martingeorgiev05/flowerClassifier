# 🌸 Flower Classifier

A full-stack web application that identifies **102 species of flowers** from a single photo. Upload any image and get back the species name and confidence score in seconds.

> **Live demo:** _[Add your Hugging Face Space link here once deployed]_

---

## Features

- Classifies **102 flower species** from the Oxford 102 Flowers dataset
- Drag-and-drop or click-to-upload interface
- Real-time image preview before submission
- Confidence score displayed as a visual percentage bar
- Input validation on both client and server (file type, size, dimensions)

---

## Tech Stack

**Machine Learning**

- TensorFlow 2.13 / Keras
- EfficientNetB5 pretrained on ImageNet (transfer learning)
- Oxford 102 Flowers dataset via `tensorflow-datasets`

**Backend**

- FastAPI — REST API with automatic OpenAPI docs
- Pillow — in-memory image processing (no uploads written to disk)
- Pydantic — request/response validation

**Frontend**

- React 19 + TypeScript
- Vite — dev server with HMR and production build

---

## Model Architecture

The classifier uses a two-phase transfer learning strategy on top of **EfficientNetB5**:

```
Input (300×300 RGB)
    │
    ▼
Data Augmentation
(flip, rotation ±30°, zoom ±30%, contrast, brightness, translation)
    │
    ▼
EfficientNetB5 backbone (ImageNet weights)
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
BatchNormalization → Dense(512, ReLU) → Dropout(0.4)
                   → Dense(256, ReLU) → Dropout(0.3)
                   → Dense(102, Softmax)
    │
    ▼
Predicted class + confidence
```

**Training phases**

| Phase           | Layers trained                          | Optimizer | Learning rate |
| --------------- | --------------------------------------- | --------- | ------------- |
| 1 — Top layers  | Classification head only (base frozen)  | AdamW     | 1e-3          |
| 2 — Fine-tuning | Last 50 layers of EfficientNetB5 + head | Adam      | 1e-5          |

Callbacks: `EarlyStopping(patience=6)`, `ModelCheckpoint` (best val_loss), `ReduceLROnPlateau(factor=0.3, patience=2)`.

---

## API Reference

**Base URL (local):** `http://localhost:8000`

### `POST /predict`

Classifies a flower image.

|                  |                       |
| ---------------- | --------------------- |
| Content-Type     | `multipart/form-data` |
| Field            | `file` — image file   |
| Accepted formats | JPEG, PNG, WEBP       |
| Max file size    | 5 MB                  |
| Max dimensions   | 8000 × 8000 px        |

**Response `200`**

```json
{
  "flower": "sunflower",
  "confidence": 97.43
}
```

**Error responses**

| Code  | Reason                                    |
| ----- | ----------------------------------------- |
| `422` | Unsupported file type or unreadable image |
| `413` | File exceeds 5 MB                         |
| `422` | Image dimensions exceed limit             |

### `GET /health`

```json
{ "status": "ok" }
```

---

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+
- A trained model at `models/flower_model.keras` (see [Training](#training))

### 1. Backend

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server (from project root)
PYTHONPATH=. uvicorn api.main:app --port 8000 --reload
```

The API is available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/predict` and `/health` to the backend automatically — no CORS setup needed.

### Quick prediction (no frontend)

```bash
python tests/test.py
# or with curl:
curl -F "file=@sample_images/daisy-test.jpg" http://localhost:8000/predict
```

---

## Training

If you want to retrain the model from scratch:

```bash
# Activate virtualenv first
python main.py
```

The training pipeline will:

1. Download the Oxford 102 Flowers dataset automatically via `tensorflow-datasets` (~330 MB, cached to `~/.keras/datasets/`)
2. Build the EfficientNetB5 model
3. Run Phase 1 training (frozen backbone, up to 30 epochs)
4. Run Phase 2 fine-tuning (last 50 layers unfrozen, up to 15 epochs)
5. Save the best checkpoint to `models/flower_model.keras`
6. Output `training_curves.png` and `confusion_matrix.png`

Hyperparameters are in [`config.py`](config.py).

---

## Project Structure

```
├── api/
│   ├── main.py              # FastAPI app, lifespan model loading, CORS
│   └── routes/
│       └── predict.py       # POST /predict + GET /health, input validation
├── frontend/
│   └── src/
│       ├── App.tsx           # UI state machine (idle → preview → loading → result)
│       ├── api/client.ts     # Typed fetch wrapper
│       └── components/
│           ├── DropZone.tsx  # Drag-and-drop upload with client-side validation
│           └── ResultCard.tsx
├── src/
│   ├── data_loader.py        # tfds dataset pipeline
│   ├── model.py              # EfficientNetB5 model definition
│   ├── train.py              # Two-phase training loop
│   ├── evaluate.py           # Metrics, training curves, confusion matrix
│   └── predict.py            # Inference (used by both API and tests)
├── config.py                 # All hyperparameters and paths
├── main.py                   # Training entry point
└── tests/test.py             # Single-image prediction script
```

---

## Deployment

This app is designed to run as a single container on **Hugging Face Spaces** (free tier).

The `Dockerfile` at the project root:

1. Builds the React frontend (`npm run build`)
2. Installs Python dependencies
3. Starts FastAPI on port `7860` (required by HF Spaces)
4. Serves both the API and the React static files from the same process

**To deploy:**

```bash
# Add HF Spaces as a git remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/flower-classifier

# Push — HF will build and deploy automatically
git push space main
```

The build takes ~5–10 minutes on first push (TensorFlow install). Subsequent pushes are faster due to Docker layer caching.

---

## Dataset

[Oxford 102 Flower Categories](https://www.tensorflow.org/datasets/catalog/oxford_flowers102) — 8,189 images across 102 flower species, split into train (1,020), validation (1,020), and test (6,149) sets.

Classes include: alpine sea holly, anthurium, artichoke, azalea, ball moss, balloon flower, barbeton daisy, bearded iris, bee balm, bird of paradise, and 92 more.

---

## License

MIT
