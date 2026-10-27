"""Exercise upload behaviour with a mocked Vision client; no cloud calls."""

import asyncio
import io
import json
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException, UploadFile
from google.cloud import vision
from PIL import Image
from starlette.datastructures import Headers

import app as api


def upload(contents=None, mime="image/png"):
    if contents is None:
        stream = io.BytesIO()
        Image.new("RGB", (320, 180), "white").save(stream, format="PNG")
        contents = stream.getvalue()
    return UploadFile(file=io.BytesIO(contents), filename="diagram.png",
                      headers=Headers({"content-type": mime}))


def response(text="Exact original OCR\n"):
    return vision.AnnotateImageResponse(
        full_text_annotation=vision.TextAnnotation(text=text, pages=[{
            "width": 320, "height": 180,
            "blocks": [{"paragraphs": [{"words": [{
                "symbols": [{"text": c} for c in "Exact"],
                "confidence": 0.95,
                "bounding_box": {"vertices": [
                    {"x": 20, "y": 20}, {"x": 80, "y": 20},
                    {"x": 80, "y": 40}, {"x": 20, "y": 40}]}
            }]}]}]
        }])
    )


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.document_text_detection.return_value = response()
        self.client_patch = patch.object(api, "get_vision_client", return_value=self.client)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

    def extract(self, file=None, include_context=False, context_mode="workflow"):
        file = file or upload()
        self.addCleanup(file.file.close)
        return api.extract_text(file=file, include_context=include_context,
                                context_mode=context_mode)

    def test_text_only_keeps_existing_fields_and_skips_visual_analysis(self):
        with patch.object(api, "analyze_visual_context") as analyzer:
            result = self.extract()
        self.assertEqual(result["filename"], "diagram.png")
        self.assertEqual(result["text"], "Exact original OCR\n")
        self.assertEqual(result["characters"], len(result["text"]))
        self.assertFalse(result.get("visual_context"))
        analyzer.assert_not_called()
        self.client.document_text_detection.assert_called_once()

    def test_context_is_added_using_the_existing_ocr_response(self):
        context = {"summary": "One text region on a white background", "regions": []}
        with patch.object(api, "analyze_visual_context", return_value=context) as analyzer:
            result = self.extract(include_context=True, context_mode="code")
        self.assertEqual(result["visual_context"], context)
        self.assertEqual(result["text"], "Exact original OCR\n")
        self.client.document_text_detection.assert_called_once()
        self.assertEqual(analyzer.call_count, 1)
        args, kwargs = analyzer.call_args
        passed_mode = kwargs.get("mode", args[2] if len(args) > 2 else None)
        self.assertEqual(passed_mode, "code")
        self.assertTrue(args[1], "Existing OCR geometry should be passed to analysis")

    def test_visual_failure_preserves_successful_ocr_and_reports_failure(self):
        with patch.object(api, "analyze_visual_context", side_effect=RuntimeError("broken analyzer")):
            result = self.extract(include_context=True)
        self.assertEqual(result["text"], "Exact original OCR\n")
        self.assertTrue(result.get("context_error"))
        self.assertFalse(result.get("visual_context"))
        self.assertNotIn("broken analyzer", result["context_error"])

    def test_invalid_image_never_reaches_google(self):
        with self.assertRaises(HTTPException) as error:
            self.extract(upload(b"not an image"))
        self.assertEqual(error.exception.status_code, 400)
        self.client.document_text_detection.assert_not_called()

    def test_unsupported_mime_never_reaches_google(self):
        with self.assertRaises(HTTPException) as error:
            self.extract(upload(mime="image/gif"))
        self.assertEqual(error.exception.status_code, 400)
        self.client.document_text_detection.assert_not_called()

    def test_large_upload_is_rejected_before_google(self):
        with self.assertRaises(HTTPException) as error:
            self.extract(upload(b"x" * (10 * 1024 * 1024 + 1)))
        self.assertEqual(error.exception.status_code, 413)
        self.client.document_text_detection.assert_not_called()

    def test_google_transport_failure_becomes_a_service_error(self):
        self.client.document_text_detection.side_effect = RuntimeError("private error details")
        with self.assertRaises(HTTPException) as error:
            self.extract()
        self.assertEqual(error.exception.status_code, 502)
        self.assertNotIn("private error details", error.exception.detail)

    def test_google_response_error_becomes_a_service_error(self):
        failed = response()
        failed.error.message = "upstream secret details"
        self.client.document_text_detection.return_value = failed
        with self.assertRaises(HTTPException) as error:
            self.extract()
        self.assertEqual(error.exception.status_code, 502)
        self.assertNotIn("upstream secret details", error.exception.detail)


async def multipart_request(fields, contents=None):
    """Call the actual ASGI upload route without an HTTP client dependency."""
    if contents is None:
        stream = io.BytesIO()
        Image.new("RGB", (320, 180), "white").save(stream, format="PNG")
        contents = stream.getvalue()
    boundary = "visual-context-offline-test"
    parts = []
    for key, value in fields.items():
        parts.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                      f'name="{key}"\r\n\r\n{value}\r\n').encode())
    parts.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                  'name="file"; filename="diagram.png"\r\n'
                  "Content-Type: image/png\r\n\r\n").encode()
                 + contents + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/ocr", "raw_path": b"/ocr",
        "root_path": "", "query_string": b"",
        "headers": [(b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
                    (b"content-length", str(len(body)).encode())],
        "server": ("testserver", 80), "client": ("127.0.0.1", 12345),
    }
    sent = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await api.app(scope, receive, send)
    status = next(message["status"] for message in sent
                  if message["type"] == "http.response.start")
    response_body = b"".join(message.get("body", b"") for message in sent
                             if message["type"] == "http.response.body")
    return status, json.loads(response_body)


class MultipartRouteTests(unittest.TestCase):
    def test_real_form_parses_context_toggle_and_mode(self):
        client = Mock()
        client.document_text_detection.return_value = response()
        context = {"summary": "Measured layout", "regions": []}
        with patch.object(api, "get_vision_client", return_value=client), \
                patch.object(api, "analyze_visual_context", return_value=context) as analyzer:
            status, payload = asyncio.run(multipart_request({
                "include_context": "true", "context_mode": "code"
            }))
        self.assertEqual(status, 200)
        self.assertEqual(payload["visual_context"], context)
        self.assertEqual(payload["text"], "Exact original OCR\n")
        args, kwargs = analyzer.call_args
        self.assertEqual(kwargs.get("mode", args[2] if len(args) > 2 else None), "code")

    def test_invalid_mode_is_rejected_before_google(self):
        with patch.object(api, "get_vision_client") as client_factory:
            status, payload = asyncio.run(multipart_request({
                "include_context": "true", "context_mode": "not-a-mode"
            }))
        self.assertEqual(status, 422)
        self.assertIn("detail", payload)
        client_factory.assert_not_called()

    def test_missing_checkbox_preserves_text_only_route(self):
        client = Mock()
        client.document_text_detection.return_value = response()
        with patch.object(api, "get_vision_client", return_value=client), \
                patch.object(api, "analyze_visual_context") as analyzer:
            status, payload = asyncio.run(multipart_request({}))
        self.assertEqual(status, 200)
        self.assertEqual(payload["text"], "Exact original OCR\n")
        analyzer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
