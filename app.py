from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import vision
from pathlib import Path

app = FastAPI(title="OCR API")

client = vision.ImageAnnotatorClient()
frontend_dist = Path(__file__).parent / "frontend" / "dist"


@app.get("/")
def home():
    index_file = frontend_dist / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {"message": "OCR API is running!"}


if (frontend_dist / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")


@app.post("/ocr")
async def extract_text(file: UploadFile = File(...)):

    # Check file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WebP images are supported"
        )

    # Read uploaded image
    content = await file.read()

    # Maximum image size: 10 MB
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image too large"
        )

    # Send image to Google Vision API
    image = vision.Image(content=content)
    try:
        response = client.document_text_detection(image=image)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="OCR service request failed"
        )

    if response.error.message:
        raise HTTPException(
            status_code=502,
            detail="OCR service returned an error"
        )

    extracted_text = response.full_text_annotation.text

    return {
        "filename": file.filename,
        "text": extracted_text,
        "characters": len(extracted_text)
    }