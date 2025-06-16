from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = tf.keras.models.load_model("aod_net_refined.keras")

# Preprocess: decode + normalize (no resizing)
def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = img.size  # (width, height) if needed
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0), original_size

# Postprocess: de-normalize and convert to PIL Image
def postprocess(output):
    output_img = (output[0] * 255).astype("uint8")
    return Image.fromarray(output_img)

# API endpoint
@app.post("/dehaze")
async def dehaze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Preprocess input
    input_tensor, _ = preprocess(image_bytes)
    
    # Run model inference
    output = model.predict(input_tensor)
    
    # Convert output to image
    result_image = postprocess(output)

    # Prepare image for streaming response
    buffer = io.BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
