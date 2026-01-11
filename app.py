from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from PIL import Image
import io
import numpy as np
import torch
from lama_cleaner.model_manager import load_model
from lama_cleaner.model import remove
import cv2

app = FastAPI()

# Load LaMa model once
device = "cuda" if torch.cuda.is_available() else "cpu"
lama_model = load_model("lama", device=device)

@app.post("/clean-frame")
async def clean_frame(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        # ---------------------------
        # Automatic mask generation
        # ---------------------------
        # Detect areas with high contrast/edges (simplified)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        mask_np = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)

        # Convert mask to PIL Image
        mask_img = Image.fromarray(mask_np).convert("L")

        # Inpaint with LaMa
        cleaned_img = remove(lama_model, img, mask=mask_img)

        # Return as PNG
        buf = io.BytesIO()
        cleaned_img.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
