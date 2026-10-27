"""Text extraction with optional local visual context analysis."""
from functools import lru_cache
from io import BytesIO
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import vision

from visual_context import MAX_IMAGE_BYTES, ImageValidationError, analyze_visual_context, decode_image, ocr_regions_from_annotation

app = FastAPI(title="Typeframe OCR and Visual Context API")
logger = logging.getLogger(__name__)
frontend_dist = Path(__file__).parent / "frontend" / "dist"


@lru_cache(maxsize=1)
def get_vision_client():
    return vision.ImageAnnotatorClient()


@app.get("/")
def home():
    index_file = frontend_dist / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {"message": "OCR API is running!"}


if (frontend_dist / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")


@app.post("/ocr")
def extract_text(file: UploadFile = File(...), include_context: bool = Form(False), context_mode: Literal["workflow", "code"] = Form("workflow")):
    # A synchronous route keeps blocking network and image work in FastAPI's worker pool.
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Only JPG, PNG and WebP images are supported")
    content = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (maximum 10 MB)")
    try:
        decoded = decode_image(content)
    except ImageValidationError as error:
        raise HTTPException(error.status_code, str(error)) from error
    # Both passes see identical dimensions, orientation and transparent background.
    normalized = BytesIO()
    decoded.save(normalized, format="PNG")
    normalized_content = normalized.getvalue()
    if len(normalized_content) > 20 * 1024 * 1024:
        raise HTTPException(413, "Decoded image is too large to process; resize it first")
    try:
        response = get_vision_client().document_text_detection(image=vision.Image(content=normalized_content), timeout=60)
    except Exception as error:
        logger.warning("OCR service request failed (%s)", type(error).__name__)
        raise HTTPException(502, "OCR service request failed") from error
    if response.error.message:
        raise HTTPException(502, "OCR service returned an error")
    annotation = response.full_text_annotation
    extracted_text = annotation.text
    result = {"filename": file.filename, "text": extracted_text, "characters": len(extracted_text)}
    if include_context:
        try:
            text_regions = ocr_regions_from_annotation(annotation, *decoded.size)
            result["visual_context"] = analyze_visual_context(decoded, text_regions, mode=context_mode)
        except Exception:
            logger.exception("Local visual context analysis failed")
            result["context_error"] = "Text was extracted, but visual context could not be analysed. Try a smaller or clearer image."
    return result
