from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras
import base64
import cv2
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class names (adjust based on your model's classes)
CLASS_NAMES = [
    "Anthracnose",
    "Bacterial_Canker",
    "Cutting_Weevil",
    "Gall_Midge",
    "Healthy",
    "Powdery_Mildew",
    "Sooty_Mould"
]

IMG_SIZE = (224, 224)

# Cache loaded models
models = {}

def load_model(model_name: str):
    if model_name in models:
        return models[model_name]
    
    model_paths = {
        "custom": "mango.keras",
        "mobilenet": "mobilenet.keras",
        "efficientnet": "efficientnet.keras"
    }
    
    path = model_paths.get(model_name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model {model_name} not found at {path}")
    
    model = keras.models.load_model(path)
    models[model_name] = model
    return model


def preprocess_image(image_bytes: bytes, model_name: str) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    
    # EfficientNet has its own preprocessing
    if model_name == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        arr = preprocess_input(arr)
    elif model_name == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0
    
    return np.expand_dims(arr, axis=0)


def get_last_conv_layer(model):
    """Find the last Conv2D layer in the model"""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # For nested models (like MobileNet/EfficientNet)
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
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


@app.get("/")
def root():
    return {"message": "Leaf Disease Classification API"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("custom")
):
    try:
        image_bytes = await file.read()
        model = load_model(model_name)
        img_array = preprocess_image(image_bytes, model_name)
        
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        disease = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else f"Class_{predicted_idx}"
        
        all_probs = {
            (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}"): float(predictions[0][i])
            for i in range(len(predictions[0]))
        }
        
        return JSONResponse({
            "success": True,
            "disease": disease,
            "accuracy": round(confidence * 100, 2),
            "model_used": model_name,
            "all_probabilities": all_probs
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/explain")
async def explain(
    file: UploadFile = File(...),
    model_name: str = Form("custom")
):
    try:
        image_bytes = await file.read()
        model = load_model(model_name)
        img_array = preprocess_image(image_bytes, model_name)
        original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        disease = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else f"Class_{predicted_idx}"
        
        last_conv = get_last_conv_layer(model)
        
        if last_conv is None:
            # Fallback: return original image
            buf = io.BytesIO()
            original_img.resize(IMG_SIZE).save(buf, format="PNG")
            overlay_b64 = base64.b64encode(buf.getvalue()).decode()
        else:
            try:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv, predicted_idx)
                overlay = overlay_gradcam(original_img, heatmap)
                buf = io.BytesIO()
                Image.fromarray(overlay).save(buf, format="PNG")
                overlay_b64 = base64.b64encode(buf.getvalue()).decode()
            except Exception as inner_e:
                buf = io.BytesIO()
                original_img.resize(IMG_SIZE).save(buf, format="PNG")
                overlay_b64 = base64.b64encode(buf.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "disease": disease,
            "accuracy": round(confidence * 100, 2),
            "model_used": model_name,
            "gradcam_image": f"data:image/png;base64,{overlay_b64}"
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)