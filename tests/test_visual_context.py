"""Offline behavioural checks for image geometry and OCR association."""

import io
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from visual_context import (
    ImageValidationError,
    analyze_visual_context,
    decode_image,
    ocr_regions_from_annotation,
)


def image_bytes(image, format="PNG", **kwargs):
    stream = io.BytesIO()
    image.save(stream, format=format, **kwargs)
    return stream.getvalue()


def text_region(id, text, bbox):
    left, top, right, bottom = bbox
    return {
        "id": id,
        "text": text,
        "bbox": list(bbox),
        "polygon": [[left, top], [right, top], [right, bottom], [left, bottom]],
        "confidence": 0.97,
    }


def two_boxes(connector=None):
    image = Image.new("RGB", (720, 340), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((30, 100, 290, 260), radius=12,
                           fill="#d9efff", outline="#202830", width=3)
    draw.rounded_rectangle((430, 100, 690, 260), radius=12,
                           fill="#dff4df", outline="#202830", width=3)
    draw.text((70, 140), "Alpha", fill="#17242d")
    draw.text((480, 140), "Beta", fill="#17242d")
    if connector == "arrow":
        pixels = np.array(image)
        cv2.arrowedLine(pixels, (292, 180), (428, 180), (30, 30, 30),
                        thickness=3, tipLength=0.14)
        image = Image.fromarray(pixels)
    elif connector == "line":
        draw.line((292, 180, 428, 180), fill="#202830", width=3)
    words = [
        text_region("t1", "Alpha", (70, 140, 160, 163)),
        text_region("t2", "Beta", (480, 140, 550, 163)),
    ]
    return image, words


def region_for_text(context, text_id):
    matches = [r for r in context["regions"] if text_id in r["text_ids"]]
    if not matches:
        raise AssertionError(f"No detected region contains OCR item {text_id}")
    return min(matches, key=lambda r: (r["bbox"][2] - r["bbox"][0]) *
               (r["bbox"][3] - r["bbox"][1]))


class DecodeImageTests(unittest.TestCase):
    def test_transparent_pixels_have_a_defined_white_background(self):
        rgba = Image.new("RGBA", (24, 20), (0, 0, 0, 0))
        decoded = decode_image(image_bytes(rgba))
        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.getpixel((0, 0)), (255, 255, 255))

    def test_exif_orientation_is_normalized(self):
        image = Image.new("RGB", (100, 60), "white")
        exif = Image.Exif()
        exif[274] = 6
        decoded = decode_image(image_bytes(image, "JPEG", exif=exif))
        self.assertEqual(decoded.size, (60, 100))
        self.assertNotIn(274, decoded.getexif())

    def test_invalid_bytes_are_rejected(self):
        with self.assertRaises(ImageValidationError) as error:
            decode_image(b"this is not a PNG")
        self.assertEqual(error.exception.status_code, 400)

    def test_decoded_pixel_limit_is_checked(self):
        with patch("visual_context.MAX_IMAGE_PIXELS", 200):
            with self.assertRaises(ImageValidationError) as error:
                decode_image(image_bytes(Image.new("RGB", (20, 20), "white")))
        self.assertEqual(error.exception.status_code, 413)

    def test_decoded_side_limit_is_checked(self):
        with patch("visual_context.MAX_IMAGE_SIDE", 20):
            with self.assertRaises(ImageValidationError) as error:
                decode_image(image_bytes(Image.new("RGB", (21, 10), "white")))
        self.assertEqual(error.exception.status_code, 413)

    def test_animation_is_not_silently_treated_as_a_single_image(self):
        stream = io.BytesIO()
        first = Image.new("RGB", (32, 32), "red")
        second = Image.new("RGB", (32, 32), "blue")
        first.save(stream, format="WEBP", save_all=True,
                   append_images=[second], duration=100, loop=0)
        with self.assertRaises(ImageValidationError):
            decode_image(stream.getvalue())


class VisualContextTests(unittest.TestCase):
    def test_coloured_containers_keep_ocr_association(self):
        image, words = two_boxes()
        context = analyze_visual_context(image, words, mode="workflow")
        alpha = region_for_text(context, "t1")
        beta = region_for_text(context, "t2")
        self.assertNotEqual(alpha["id"], beta["id"])
        self.assertEqual(context["image"], {"width": 720, "height": 340})
        self.assertEqual(alpha["row"], beta["row"])
        self.assertLess(alpha["column"], beta["column"])
        for region, expected in [(alpha, (217, 239, 255)), (beta, (223, 244, 223))]:
            observed = tuple(int(region["fill"]["hex"].lstrip("#")[i:i+2], 16)
                             for i in (0, 2, 4))
            self.assertLess(sum(abs(a-b) for a, b in zip(observed, expected)), 65)
        self.assertTrue(context["summary"].strip())

    def test_box_borders_do_not_invent_links(self):
        image, words = two_boxes()
        context = analyze_visual_context(image, words, mode="workflow")
        self.assertEqual(context["connections"], [])

    def test_heading_rule_does_not_split_one_card_into_flow_nodes(self):
        image = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((80, 65, 520, 335), radius=14,
                               fill="#d9efff", outline="#202830", width=3)
        draw.line((82, 115, 518, 115), fill="#202830", width=3)
        draw.text((110, 83), "HEADER", fill="#17242d")
        draw.text((130, 180), "Body item", fill="#17242d")
        words = [text_region("heading", "HEADER", (110, 83, 205, 103)),
                 text_region("body", "Body item", (130, 180, 250, 203))]

        context = analyze_visual_context(image, words, mode="workflow")
        heading = region_for_text(context, "heading")
        body = region_for_text(context, "body")
        self.assertEqual(heading["id"], body["id"],
                         "A heading separator belongs to the same card")
        primary = [region for region in context["regions"]
                   if region["parent_id"] is None and region["shape"] != "text_group"]
        self.assertEqual(len(primary), 1)
        self.assertEqual(context["connections"], [])

    def test_bent_connector_attaches_to_terminals_not_nearby_card(self):
        image = Image.new("RGB", (920, 540), "white")
        draw = ImageDraw.Draw(image)
        for bounds, colour in [((30, 80, 230, 220), "#d9efff"),
                               ((650, 320, 870, 470), "#dff4df"),
                               ((315, 110, 515, 245), "#ffe8be")]:
            draw.rectangle(bounds, fill=colour, outline="#202830", width=3)
        draw.text((60, 140), "Start", fill="#17242d")
        draw.text((680, 370), "Finish", fill="#17242d")
        draw.text((350, 165), "Unrelated", fill="#17242d")
        # The vertical segment passes only 17 px from the unrelated card.
        # Its true endpoints meet Start's right side and Finish's top side.
        draw.line([(232, 150), (298, 150), (298, 280), (760, 280), (760, 317)],
                  fill="#202830", width=3)
        words = [text_region("start", "Start", (60, 140, 140, 163)),
                 text_region("finish", "Finish", (680, 370, 770, 393)),
                 text_region("other", "Unrelated", (350, 165, 470, 188))]

        context = analyze_visual_context(image, words, mode="workflow")
        start = region_for_text(context, "start")["id"]
        finish = region_for_text(context, "finish")["id"]
        other = region_for_text(context, "other")["id"]
        expected = [edge for edge in context["connections"]
                    if {edge["source"], edge["target"]} == {start, finish}]
        self.assertTrue(expected, "The bent connector must retain its real endpoints")
        self.assertFalse(any(other in (edge["source"], edge["target"])
                             for edge in context["connections"]),
                         "Passing near a card must not create a connection to it")
        self.assertTrue(all(edge["direction"] == "unknown" for edge in expected))

    def test_analysis_resize_preserves_original_image_coordinates(self):
        base, words = two_boxes()
        scale = 4
        image = base.resize((base.width * scale, base.height * scale),
                            Image.Resampling.NEAREST)
        original_words = [text_region(word["id"], word["text"],
                                      [value * scale for value in word["bbox"]])
                          for word in words]

        context = analyze_visual_context(image, original_words, mode="workflow")
        self.assertEqual(context["image"], {"width": 2880, "height": 1360})
        for word, expected in [("t1", (120, 400, 1160, 1040)),
                               ("t2", (1720, 400, 2760, 1040))]:
            box = region_for_text(context, word)["bbox"]
            for observed, target in zip(box, expected):
                self.assertAlmostEqual(observed, target, delta=16,
                                       msg="Analysis-size coordinates leaked into the export")
        self.assertEqual([region["bbox"] for region in context["text_regions"]],
                         [region["bbox"] for region in original_words])

    def test_pastel_green_container_is_not_described_as_white(self):
        image, words = two_boxes()
        context = analyze_visual_context(image, words, mode="workflow")
        beta = region_for_text(context, "t2")
        self.assertIn("green", beta["fill"]["name"])

    def test_decision_diamond_is_a_container(self):
        image = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(image)
        draw.polygon([(300, 50), (420, 170), (300, 290), (180, 170)],
                     fill="#d9efff", outline="#202830", width=3)
        draw.text((250, 160), "Ready?", fill="#17242d")
        words = [text_region("decision", "Ready?", (250, 160, 350, 180))]
        context = analyze_visual_context(image, words, mode="workflow")
        decision = region_for_text(context, "decision")
        self.assertEqual(decision["shape"], "diamond")

    def test_parent_container_does_not_erase_child_connector(self):
        image = Image.new("RGB", (720, 400), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 20, 710, 380), fill="#eeeeee",
                       outline="#202830", width=3)
        draw.rectangle((40, 130, 280, 280), fill="#d9efff",
                       outline="#202830", width=3)
        draw.rectangle((440, 130, 680, 280), fill="#dff4df",
                       outline="#202830", width=3)
        draw.text((80, 165), "Alpha", fill="#17242d")
        draw.text((480, 165), "Beta", fill="#17242d")
        pixels = np.array(image)
        cv2.arrowedLine(pixels, (283, 205), (437, 205), (20, 20, 20),
                        thickness=3, tipLength=0.12)
        words = [text_region("t1", "Alpha", (80, 165, 150, 185)),
                 text_region("t2", "Beta", (480, 165, 540, 185))]
        context = analyze_visual_context(Image.fromarray(pixels), words,
                                         mode="workflow")
        alpha = region_for_text(context, "t1")
        beta = region_for_text(context, "t2")
        self.assertIsNotNone(alpha["parent_id"])
        self.assertEqual(alpha["parent_id"], beta["parent_id"])
        self.assertEqual(alpha["row"], beta["row"])
        self.assertEqual((alpha["column"], beta["column"]), (1, 2))
        edges = [edge for edge in context["connections"]
                 if {edge["source"], edge["target"]} == {alpha["id"], beta["id"]}]
        self.assertTrue(edges, "An enclosing panel must not erase its child connector")
        self.assertTrue(any(
            (edge["source"] == alpha["id"] and edge["direction"] == "source_to_target")
            or (edge["source"] == beta["id"] and edge["direction"] == "target_to_source")
            for edge in edges
        ), "The child arrow should retain its direction")

    def test_parent_fill_excludes_large_child_container(self):
        image = Image.new("RGB", (800, 500), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, 760, 460), fill="#92b6ff",
                       outline="#202830", width=3)
        draw.rectangle((75, 75, 725, 425), fill="#ffce62",
                       outline="#202830", width=3)
        context = analyze_visual_context(image, [], mode="workflow")
        children = [region for region in context["regions"] if region["parent_id"]]
        self.assertTrue(children, "A legitimate large child should remain separate")
        child = max(children, key=lambda region: (region["bbox"][2] - region["bbox"][0]) *
                    (region["bbox"][3] - region["bbox"][1]))
        parent = next(region for region in context["regions"]
                      if region["id"] == child["parent_id"])
        observed = tuple(int(parent["fill"]["hex"].lstrip("#")[i:i+2], 16)
                         for i in (0, 2, 4))
        self.assertLess(sum(abs(a-b) for a, b in zip(observed, (146, 182, 255))), 65,
                        "Parent background must not inherit its child's orange fill")

    def test_clear_arrow_connects_the_two_containers(self):
        image, words = two_boxes("arrow")
        context = analyze_visual_context(image, words, mode="workflow")
        alpha = region_for_text(context, "t1")["id"]
        beta = region_for_text(context, "t2")["id"]
        edges = [edge for edge in context["connections"]
                 if {edge["source"], edge["target"]} == {alpha, beta}]
        self.assertTrue(edges, "The visible connector must be represented")
        confirmed = [edge for edge in edges if edge["direction"] != "unknown"]
        self.assertTrue(confirmed, "A clean, explicit arrowhead should establish direction")
        for edge in confirmed:
            correct = ((edge["source"] == alpha and edge["direction"] == "source_to_target")
                       or (edge["source"] == beta and edge["direction"] == "target_to_source"))
            self.assertTrue(correct, f"Arrow direction was reversed: {edge}")

    def test_a_plain_line_has_no_invented_arrowhead(self):
        image, words = two_boxes("line")
        context = analyze_visual_context(image, words, mode="workflow")
        alpha = region_for_text(context, "t1")["id"]
        beta = region_for_text(context, "t2")["id"]
        edges = [edge for edge in context["connections"]
                 if {edge["source"], edge["target"]} == {alpha, beta}]
        self.assertTrue(edges, "A visible undirected connector should not disappear")
        self.assertTrue(all(edge["direction"] == "unknown" for edge in edges))
        self.assertTrue(context["uncertainties"])

    def test_solid_images_do_not_invent_structure(self):
        for colour in ("white", "#15191e"):
            with self.subTest(colour=colour):
                context = analyze_visual_context(Image.new("RGB", (420, 300), colour), [],
                                                 mode="workflow")
                self.assertEqual(context["regions"], [])
                self.assertEqual(context["connections"], [])
                self.assertEqual(context["text_regions"], [])
                self.assertTrue(context["background"]["hex"].startswith("#"))

    def test_code_line_order_and_relative_indentation(self):
        image = Image.new("RGB", (500, 210), "#17202a")
        draw = ImageDraw.Draw(image)
        lines = [
            text_region("t1", "if ready:", (40, 40, 145, 60)),
            text_region("t2", "run_job()", (80, 85, 195, 105)),
            text_region("t3", "save_result()", (80, 130, 240, 150)),
        ]
        for item in lines:
            draw.text(tuple(item["bbox"][:2]), item["text"], fill="#e8eef7")
        context = analyze_visual_context(image, lines, mode="code")
        layout = context["code_layout"]
        self.assertIsNotNone(layout)
        self.assertEqual(layout["tabs_or_spaces"], "unknown")
        self.assertEqual(len(layout["lines"]), 3)
        self.assertEqual([line["text_ids"] for line in layout["lines"]],
                         [["t1"], ["t2"], ["t3"]])
        self.assertAlmostEqual(layout["lines"][0]["indent_px"], 0, delta=2)
        self.assertAlmostEqual(layout["lines"][1]["indent_px"], 40, delta=2)
        self.assertEqual(layout["lines"][1]["indent_level"],
                         layout["lines"][2]["indent_level"])
        self.assertGreater(layout["lines"][1]["indent_level"],
                           layout["lines"][0]["indent_level"])


class OCRGeometryTests(unittest.TestCase):
    def test_word_text_and_polygon_survive_annotation_conversion(self):
        from google.cloud import vision

        annotation = vision.TextAnnotation(
            text="SHAP Δ",
            pages=[{
                "width": 500, "height": 300,
                "blocks": [{"paragraphs": [{"words": [
                    {"symbols": [{"text": c} for c in "SHAP"],
                     "confidence": 0.96,
                     "bounding_box": {"vertices": [
                         {"x": 40, "y": 50}, {"x": 100, "y": 50},
                         {"x": 100, "y": 70}, {"x": 40, "y": 70}]}},
                    {"symbols": [{"text": "Δ"}], "confidence": 0.88,
                     "bounding_box": {"vertices": [
                         {"x": 110, "y": 50}, {"x": 130, "y": 50},
                         {"x": 130, "y": 70}, {"x": 110, "y": 70}]}}
                ]}]}]
            }]
        )
        regions = ocr_regions_from_annotation(annotation, 500, 300)
        self.assertEqual([r["text"] for r in regions], ["SHAP", "Δ"])
        self.assertEqual(regions[0]["bbox"], [40, 50, 100, 70])
        self.assertEqual(len(regions[0]["polygon"]), 4)
        self.assertAlmostEqual(regions[0]["confidence"], 0.96, places=4)
        self.assertNotEqual(regions[0]["id"], regions[1]["id"])


if __name__ == "__main__":
    unittest.main()
