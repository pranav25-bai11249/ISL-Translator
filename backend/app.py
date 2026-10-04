"""
ISL Real-Time Translator - FastAPI backend.

Loads the trained CNN + LSTM model (models/best_isl_model.keras or isl_model.keras)
together with label_map.json. If no model is found, it falls back to a motion
heuristic and marks every response with source="heuristic" so it can't be mistaken
for a real result.

Run (from the backend folder):
    pip install fastapi uvicorn pydantic
    python -m uvicorn app:app --reload --port 8000

Model lookup order:
    1. folder in the ISL_MODEL_DIR environment variable
    2. ./models
    3. ../src/models
"""

import os
import json
import time
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Contract - must match data_collection.py, model.py and the frontend
# --------------------------------------------------------------------------

FRAME_WINDOW = 30    # frames per sign sample
FEATURE_DIM = 258    # 132 pose + 63 left hand + 63 right hand
POSE_DIM = 132       # 33 pose landmarks x (x, y, z, visibility)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

app = FastAPI(
    title="ISL Real-Time Translator",
    version="0.2.0",
    description="Landmark window in, sign label out.",
)

# When deployed, set ALLOWED_ORIGINS to your frontend address(es), comma separated,
# for example: https://my-isl-app.vercel.app   (no trailing slash)
EXTRA_ORIGINS = [
    o.strip().rstrip("/")
    for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS + EXTRA_ORIGINS,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",  # any local port, for development
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------------------------------
# Request / response shapes
# --------------------------------------------------------------------------

class FrameData(BaseModel):
    """Body of POST /predict - a (30, 258) window of landmark floats."""
    landmarks: List[List[float]]


class Prediction(BaseModel):
    prediction: str
    status: str
    confidence: float
    inference_ms: float
    source: str


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def find_model_files():
    candidates = [
        os.environ.get("ISL_MODEL_DIR"),
        os.path.join(BASE_DIR, "models"),
        os.path.join(BASE_DIR, "..", "src", "models"),
    ]
    for folder in candidates:
        if not folder:
            continue
        labels = os.path.join(folder, "label_map.json")
        if not os.path.exists(labels):
            continue
        for name in ("best_isl_model.keras", "isl_model.keras"):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return os.path.abspath(path), os.path.abspath(labels)
    return None, None


def load_model():
    """Returns (model, labels_dict). (None, {}) if nothing usable is found."""
    model_path, label_path = find_model_files()
    if model_path is None:
        print("[ISL] No trained model found. Using heuristic fallback.")
        return None, {}
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        with open(label_path, encoding="utf-8") as f:
            labels = {int(k): v for k, v in json.load(f).items()}
        print(f"[ISL] Loaded model: {model_path}")
        print(f"[ISL] Labels: {labels}")
        return model, labels
    except Exception as exc:  # TensorFlow missing, bad file, shape mismatch...
        print(f"[ISL] Could not load model ({exc}). Using heuristic fallback.")
        return None, {}


model, LABELS = load_model()


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def classify(window: List[List[float]]):
    """Return (label, confidence)."""
    if model is not None:
        x = np.asarray(window, dtype=np.float32)[np.newaxis, ...]   # (1, 30, 258)
        probs = model.predict(x, verbose=0)[0]
        i = int(np.argmax(probs))
        return LABELS.get(i, f"class_{i}"), float(probs[i])
    return heuristic(window)


def heuristic(window: List[List[float]]):
    """
    Fallback only. Mean hand displacement across the window plus mean hand height
    in the last frame. Not a trained model.
    """
    hand_dims = FEATURE_DIM - POSE_DIM

    motion = 0.0
    for f in range(1, len(window)):
        prev, cur = window[f - 1], window[f]
        for i in range(POSE_DIM, FEATURE_DIM):
            motion += abs(cur[i] - prev[i])
    motion /= max(1, (len(window) - 1) * hand_dims)

    last = window[-1]
    ys = [last[i] for i in range(POSE_DIM + 1, FEATURE_DIM, 3) if last[i] != 0]
    height = sum(ys) / len(ys) if ys else 0.5

    if motion > 0.012:
        label = "hello"
    elif height < 0.45:
        label = "iloveyou"
    else:
        label = "thanks"

    confidence = min(0.99, 0.62 + min(motion * 18, 0.3))
    return label, confidence


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

def root():
    return {
        "service": "ISL Real-Time Translator",
        "predict": "POST /predict",
        "expects": f"({FRAME_WINDOW}, {FEATURE_DIM}) landmark window",
        "model_loaded": model is not None,
        "labels": list(LABELS.values()),
    }


STATIC_DIR = os.path.join(BASE_DIR, "static")   # the built frontend, present only in the single-platform Docker image
HAS_FRONTEND = os.path.isdir(STATIC_DIR)

app.add_api_route("/api", root, methods=["GET"])
if not HAS_FRONTEND:
    app.add_api_route("/", root, methods=["GET"])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=Prediction)
def predict_sign(data: FrameData):
    """Landmark window in, sign label out. Wrong shapes are rejected with a 400."""
    window = data.landmarks

    if len(window) != FRAME_WINDOW:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {FRAME_WINDOW} frames, got {len(window)}.",
        )

    for n, frame in enumerate(window):
        if len(frame) != FEATURE_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Frame {n} has {len(frame)} features, expected {FEATURE_DIM}.",
            )

    started = time.perf_counter()
    label, confidence = classify(window)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return Prediction(
        prediction=label,
        status="success",
        confidence=round(confidence, 4),
        inference_ms=round(elapsed_ms, 2),
        source="model" if model is not None else "heuristic",
    )


# Must stay LAST: it serves the website for every path that is not an API route above.
if HAS_FRONTEND:
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")