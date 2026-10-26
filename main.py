from google.cloud import vision

# Create Google Vision client
client = vision.ImageAnnotatorClient()

def ocr_image(path):
    # Read image
    with open(path, "rb") as f:
        content = f.read()

    # Convert image into Vision API format
    image = vision.Image(content=content)

    # Send image to Google Cloud Vision
    response = client.document_text_detection(image=image)

    # Check for API errors
    if response.error.message:
        raise Exception(response.error.message)

    # Return extracted text
    return response.full_text_annotation.text


# Test OCR locally
if __name__ == "__main__":
    text = ocr_image("snta.png")
    print("Extracted Text:")
    print(text)