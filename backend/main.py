import os
# Uncomment if your .keras was saved by Keras 2 (TF 2.15 era):
# os.environ["TF_USE_LEGACY_KERAS"] = "1"

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import tensorflow as tf
import keras
import cv2

# =============================================================================
# APP
# =============================================================================
app = FastAPI(title="Mango Leaf Disease Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Same file used for all three keys (temporary — until real models exist)
SAME_MODEL = MODELS_DIR / "mango_leaf_cnn.keras"

MODEL_PATHS = {
    "custom":       SAME_MODEL,
    "mobilenet":    SAME_MODEL,
    "efficientnet": SAME_MODEL,
}

# Class names in the exact index order your model was trained on
CLASS_NAMES = [
    "Anthracnose",      # 0
    "Bacterial Canker", # 1
    "Cutting Weevil",   # 2
    "Die Back",         # 3
    "Gall Midge",       # 4
    "Healthy",          # 5
    "Powdery Mildew",   # 6
    "Sooty Mould",      # 7
]

IMG_SIZE = (224, 224)

# Cache — since all 3 keys point to the same file, we load it only once
models = {}


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model(model_name: str):
    if model_name in models:
        return models[model_name]

    path = MODEL_PATHS.get(model_name)
    if path is None:
        raise ValueError(f"Unknown model: {model_name}")

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Reuse the already-loaded model when paths match (same file)
    for cached_name, cached_model in models.items():
        if MODEL_PATHS.get(cached_name) == path:
            models[model_name] = cached_model
            print(f"♻️  Reusing already-loaded model for '{model_name}'")
            return cached_model

    model = keras.models.load_model(str(path))
    models[model_name] = model
    print(f"✅ Loaded '{model_name}' from {path}")
    return model


# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess_image(image_bytes: bytes, model_name: str) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)

    # Since the same custom model is used for all keys, always use 1/255 rescale.
    # When you swap in real MobileNet/EfficientNet models, re-enable the branches
    # below — they are commented so predictions stay correct for the custom model.
    #
    # if model_name == "mobilenet":
    #     from keras.applications.mobilenet_v2 import preprocess_input
    #     arr = preprocess_input(arr)
    # elif model_name == "efficientnet":
    #     from keras.applications.efficientnet import preprocess_input
    #     arr = preprocess_input(arr)
    # else:
    arr = arr / 255.0

    return np.expand_dims(arr, axis=0)


# =============================================================================
# GRAD-CAM HELPERS
# =============================================================================
def get_last_conv_layer(model):
    """Return the name of the last Conv2D layer (handles wrapped sub-models)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img: Image.Image, heatmap: np.ndarray, alpha=0.4):
    img = np.array(original_img.resize(IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    superimposed = jet * alpha + img * (1 - alpha)
    return np.uint8(superimposed)


def _image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# =============================================================================
# ROUTES
# =============================================================================
@app.get("/")
def root():
    return {
        "message": "Mango Leaf Disease Classifier API",
        "models": list(MODEL_PATHS.keys()),
        "note": "All 3 keys currently point to the same model file.",
        "classes": CLASS_NAMES,
        "img_size": list(IMG_SIZE),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("custom"),
):
    try:
        image_bytes = await file.read()
        model = load_model(model_name)
        img_array = preprocess_image(image_bytes, model_name)

        predictions = model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        disease = (
            CLASS_NAMES[predicted_idx]
            if predicted_idx < len(CLASS_NAMES)
            else f"Class_{predicted_idx}"
        )

        all_probs = {
            (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}"): float(
                predictions[0][i]
            )
            for i in range(len(predictions[0]))
        }

        return JSONResponse({
            "success": True,
            "disease": disease,
            "accuracy": round(confidence * 100, 2),
            "model_used": model_name,
            "all_probabilities": all_probs,
        })

    except FileNotFoundError as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/explain")
async def explain(
    file: UploadFile = File(...),
    model_name: str = Form("custom"),
):
    try:
        image_bytes = await file.read()
        model = load_model(model_name)
        img_array = preprocess_image(image_bytes, model_name)
        original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        predictions = model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        disease = (
            CLASS_NAMES[predicted_idx]
            if predicted_idx < len(CLASS_NAMES)
            else f"Class_{predicted_idx}"
        )

        last_conv = get_last_conv_layer(model)

        if last_conv is None:
            overlay_b64 = _image_to_b64(original_img.resize(IMG_SIZE))
        else:
            try:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv, predicted_idx)
                overlay = overlay_gradcam(original_img, heatmap)
                overlay_b64 = _image_to_b64(Image.fromarray(overlay))
            except Exception as inner_e:
                print(f"⚠️  Grad-CAM failed for {model_name}: {inner_e}")
                overlay_b64 = _image_to_b64(original_img.resize(IMG_SIZE))

        return JSONResponse({
            "success": True,
            "disease": disease,
            "accuracy": round(confidence * 100, 2),
            "model_used": model_name,
            "gradcam_image": f"data:image/png;base64,{overlay_b64}",
        })

    except FileNotFoundError as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)