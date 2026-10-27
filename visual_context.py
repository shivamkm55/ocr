"""Measured visual context. OCR supplies text; unclear geometry stays uncertain."""
from collections import defaultdict, deque
import colorsys
from io import BytesIO
import math
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIDE = 10_000
MAX_ANALYSIS_SIDE = 1400
MAX_TEXT_REGIONS = 8000
MAX_REGIONS = 60


class ImageValidationError(ValueError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def decode_image(content):
    if len(content) > MAX_IMAGE_BYTES:
        raise ImageValidationError("Image too large (maximum 10 MB)", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ImageValidationError("Only JPG, PNG and WebP images are supported")
                if source.width * source.height > MAX_IMAGE_PIXELS or max(source.size) > MAX_IMAGE_SIDE:
                    raise ImageValidationError("Image dimensions are too large (maximum 20 megapixels and 10000 pixels per side)", 413)
                if getattr(source, "is_animated", False):
                    raise ImageValidationError("Please upload a still image instead of an animation")
                source.load()
                oriented = ImageOps.exif_transpose(source)
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
                return oriented.convert("RGB")
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageValidationError("Image dimensions are too large", 413) from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageValidationError("The uploaded file is not a readable image") from error


def ocr_regions_from_annotation(annotation, width, height):
    """Reuse Vision word text and polygons without another OCR call."""
    output = []
    for pi, page in enumerate(annotation.pages):
        scale_x = width / (getattr(page, "width", 0) or width)
        scale_y = height / (getattr(page, "height", 0) or height)
        for bi, block in enumerate(page.blocks):
            for qi, paragraph in enumerate(block.paragraphs):
                for word in paragraph.words:
                    text = "".join(symbol.text for symbol in word.symbols)
                    if not text:
                        continue
                    bounds = word.bounding_box
                    polygon = [[round(v.x * scale_x), round(v.y * scale_y)] for v in bounds.vertices] if len(bounds.vertices) else [[round(v.x * width), round(v.y * height)] for v in bounds.normalized_vertices]
                    if len(polygon) < 3:
                        continue
                    polygon = [[max(0, min(width, x)), max(0, min(height, y))] for x, y in polygon]
                    xs, ys = zip(*polygon)
                    if max(xs) <= min(xs) or max(ys) <= min(ys):
                        continue
                    output.append({"id": f"t{len(output) + 1}", "text": text, "polygon": polygon, "bbox": [min(xs), min(ys), max(xs), max(ys)], "confidence": round(float(getattr(word, "confidence", 0)), 3), "block_id": f"p{pi + 1}_b{bi + 1}", "paragraph_id": f"p{pi + 1}_b{bi + 1}_p{qi + 1}"})
                    if len(output) >= MAX_TEXT_REGIONS:
                        return output
    return output


def _color(rgb):
    r, g, b = [int(x) for x in rgb]
    hue, saturation, value = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if saturation < 0.05 or max(r, g, b) - min(r, g, b) < 12:
        name = "white" if value > 0.97 else "light grey" if value > 0.68 else "grey" if value > 0.32 else "dark grey" if value > 0.12 else "black"
    else:
        angle = hue * 360
        name = "red" if angle < 15 or angle >= 345 else "orange" if angle < 45 else "yellow" if angle < 70 else "green" if angle < 155 else "teal" if angle < 190 else "blue" if angle < 255 else "purple" if angle < 295 else "pink"
        if min(r, g, b) > 135:
            name = "light " + name
        elif value < 0.48:
            name = "dark " + name
    return {"hex": f"#{r:02x}{g:02x}{b:02x}", "name": name}


def _dominant(pixels):
    flat = np.asarray(pixels).reshape(-1, 3)
    if not len(flat):
        return np.array([255, 255, 255], dtype=np.uint8)
    flat = flat[::max(1, len(flat) // 30000)]
    _, inverse, counts = np.unique(flat // 24, axis=0, return_inverse=True, return_counts=True)
    return np.median(flat[inverse == int(np.argmax(counts))], axis=0).astype(np.uint8)


def _palette(rgb, mask):
    step_y, step_x = max(1, rgb.shape[0] // 200), max(1, rgb.shape[1] // 200)
    sample = rgb[::step_y, ::step_x]
    pixels = sample[mask[::step_y, ::step_x] == 0]
    if not len(pixels):
        pixels = sample.reshape(-1, 3)
    _, inverse, counts = np.unique(pixels // 32, axis=0, return_inverse=True, return_counts=True)
    return [{**_color(np.median(pixels[inverse == i], axis=0)), "fraction": round(float(counts[i] / len(pixels)), 3)} for i in np.argsort(counts)[::-1][:6] if counts[i] / len(pixels) >= 0.012]


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a, b):
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def _iou(a, b):
    overlap = _intersection(a, b)
    return overlap / max(1, _area(a) + _area(b) - overlap)


def _text_mask(shape, text_regions, sx=1, sy=1):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for region in text_regions:
        l, t, r, b = region["bbox"]
        polygon = region.get("polygon") or [[l, t], [r, t], [r, b], [l, b]]
        points = np.array([[round(x / sx), round(y / sy)] for x, y in polygon], np.int32)
        cv2.fillPoly(mask, [points], 255)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def _find_regions(rgb, text_mask, background, text_regions, sx, sy):
    height, width = rgb.shape[:2]
    edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 50, 130)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    colored = ((hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 65) & (distance > 28)).astype(np.uint8) * 255
    colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    candidates = []
    for binary in (edges, colored):
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if w < 35 or h < 24 or w * h < max(650, width * height * 0.006) or w * h > width * height * 0.94 or w / h > 18 or h / w > 18 or area / (w * h) < 0.40:
                continue
            approx = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
            if not cv2.isContourConvex(approx) or not 4 <= len(approx) <= 12:
                continue
            shape = "rectangle"
            if len(approx) == 4:
                midpoints = ((x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2))
                distances = [min(np.linalg.norm(v - c) for c in midpoints) for v in approx.reshape(-1, 2)]
                if max(distances) < min(w, h) * 0.22:
                    shape = "diamond"
            elif area / (w * h) < 0.85:
                shape = "ellipse_or_rounded_shape"
            else:
                shape = "rounded_rectangle"
            candidates.append({"bounds": [x, y, x + w, y + h], "shape": shape})
    kept = []
    for candidate in sorted(candidates, key=lambda c: _area(c["bounds"]), reverse=True):
        if not any(_iou(candidate["bounds"], other["bounds"]) > 0.86 for other in kept):
            kept.append(candidate)
        if len(kept) >= MAX_REGIONS:
            break
    # Header rules and body outlines can each create another full-width contour
    # within one card. Preserve their appearance as bands, not extra flow nodes.
    bands = []
    filtered = []
    for candidate in kept:
        box = candidate["bounds"]
        partitions = []
        for parent in kept:
            p = parent["bounds"]
            expanded = [p[0] - 5, p[1] - 5, p[2] + 5, p[3] + 5]
            if parent is candidate or _area(p) < _area(box) * 1.15:
                continue
            same_width = abs(box[0] - p[0]) <= 6 and abs(box[2] - p[2]) <= 6
            at_edge = abs(box[1] - p[1]) <= 6 or abs(box[3] - p[3]) <= 6
            if same_width and at_edge and _intersection(expanded, box) / max(1, _area(box)) > 0.97:
                partitions.append(parent)
        if partitions:
            parent = min(partitions, key=lambda p: _area(p["bounds"]))
            if box[3] - box[1] < (parent["bounds"][3] - parent["bounds"][1]) * 0.4:
                bands.append((candidate, parent["bounds"]))
        else:
            filtered.append(candidate)
    kept = filtered
    regions = []
    for candidate in sorted(kept, key=lambda c: (c["bounds"][1], c["bounds"][0])):
        l, t, r, b = candidate["bounds"]
        margin = min(8, max(3, min(r - l, b - t) // 40))
        interior = rgb[t + margin:b - margin, l + margin:r - margin]
        available = (text_mask[t + margin:b - margin, l + margin:r - margin] == 0).astype(np.uint8)
        for child in kept:
            box = child["bounds"]
            if child is candidate or _area(box) >= _area(candidate["bounds"]) * 0.9:
                continue
            if _intersection(candidate["bounds"], box) / max(1, _area(box)) > 0.97:
                cv2.rectangle(available, (box[0] - l - margin, box[1] - t - margin), (box[2] - l - margin, box[3] - t - margin), 0, -1)
        pixels = interior[available != 0]
        appearance_bands = []
        for band, parent_bounds in bands:
            if _iou(candidate["bounds"], parent_bounds) > 0.9:
                bl, bt, br, bb = band["bounds"]
                patch = rgb[bt + 2:bb - 2, bl + 2:br - 2]
                appearance_bands.append({"bbox": [round(bl * sx), round(bt * sy), round(br * sx), round(bb * sy)], "fill": _color(_dominant(patch))})
        regions.append({"id": f"r{len(regions) + 1}", "bbox": [round(l * sx), round(t * sy), round(r * sx), round(b * sy)], "shape": candidate["shape"], "fill": _color(_dominant(pixels if len(pixels) else interior)), "parent_id": None, "text_ids": [], "appearance_bands": appearance_bands})
    if not regions and text_regions:
        groups = defaultdict(list)
        for text in text_regions:
            groups[text.get("block_id", "text_block")].append(text)
        for texts in list(groups.values())[:MAX_REGIONS]:
            boxes = [text["bbox"] for text in texts]
            bounds = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
            regions.append({"id": f"r{len(regions) + 1}", "bbox": bounds, "shape": "text_group", "fill": None, "parent_id": None, "text_ids": []})
    for region in regions:
        parents = [other for other in regions if other is not region and _area(other["bbox"]) > _area(region["bbox"]) * 1.15 and _intersection([other["bbox"][0] - 5 * sx, other["bbox"][1] - 5 * sy, other["bbox"][2] + 5 * sx, other["bbox"][3] + 5 * sy], region["bbox"]) / max(1, _area(region["bbox"])) > 0.97]
        if parents:
            region["parent_id"] = min(parents, key=lambda p: _area(p["bbox"]))["id"]
    for text in text_regions:
        matches = [r for r in regions if _intersection(r["bbox"], text["bbox"]) / max(1, _area(text["bbox"])) > 0.6]
        if matches:
            min(matches, key=lambda r: _area(r["bbox"]))["text_ids"].append(text["id"])
    siblings = defaultdict(list)
    for region in regions:
        siblings[region["parent_id"]].append(region)
    for group in siblings.values():
        rows = []
        for region in sorted(group, key=lambda r: (r["bbox"][1], r["bbox"][0])):
            box = region["bbox"]
            center = (box[1] + box[3]) / 2
            row = next((row for row in rows if abs(row["center"] - center) < min(row["height"], box[3] - box[1]) * 0.45), None)
            if row is None:
                row = {"center": center, "height": box[3] - box[1], "members": []}
                rows.append(row)
            row["members"].append(region)
        for row_index, row in enumerate(sorted(rows, key=lambda row: row["center"]), 1):
            for column, region in enumerate(sorted(row["members"], key=lambda r: r["bbox"][0]), 1):
                region.update(row=row_index, column=column)
    return regions


def _distance_to_box(points, box):
    x, y = points[:, 0], points[:, 1]
    dx = np.maximum(np.maximum(box[0] - x, 0), x - box[2])
    dy = np.maximum(np.maximum(box[1] - y, 0), y - box[3])
    distances = dx * dx + dy * dy
    inside = (dx == 0) & (dy == 0)
    boundary = np.minimum.reduce([x - box[0], box[2] - x, y - box[1], box[3] - y])
    distances[inside] = boundary[inside] ** 2
    index = int(np.argmin(distances))
    return math.sqrt(float(distances[index])), index


def _path_in_component(mask, start, end):
    """Trace connected pixels instead of inventing a straight shortcut."""
    start, end = tuple(map(int, start)), tuple(map(int, end))
    queue = deque([start])
    previous = {start: None}
    height, width = mask.shape
    while queue:
        current = queue.popleft()
        if current == end:
            break
        x, y = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nxt = (x + dx, y + dy)
            if 0 <= nxt[0] < width and 0 <= nxt[1] < height and mask[nxt[1], nxt[0]] and nxt not in previous:
                previous[nxt] = current
                queue.append(nxt)
    if end not in previous:
        return [list(start), list(end)]
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    simplified = cv2.approxPolyDP(np.array(path[::-1], np.int32).reshape(-1, 1, 2), 1.8, False)
    return simplified.reshape(-1, 2).tolist()


def _arrow_at(points, toward, span):
    """Require two wings widening behind a narrow tip and shaft."""
    norm = float(np.linalg.norm(toward))
    if norm < 1:
        return False
    axis = toward / norm
    side = np.array([-axis[1], axis[0]])
    tip = points[int(np.argmax(points @ axis))]
    behind = (tip - points) @ axis
    across = (points - tip) @ side
    window = min(26, max(11, span * 0.18))
    wings = across[(behind > 1.5) & (behind < window * 0.75)]
    shaft = across[(behind > window) & (behind < window * 1.8)]
    if len(wings) < 6 or len(shaft) < 3:
        return False
    shaft_width = float(np.percentile(np.abs(shaft), 85))
    return min(float(np.max(wings)), float(-np.min(wings))) > max(3, shaft_width * 1.7 + 0.7)


def _skeleton_endpoints(mask):
    """Zhang-Suen thinning keeps bends, avoiding attachment to a nearby mid-line box."""
    thin = (mask != 0).astype(np.uint8)
    for _ in range(100):
        changed = False
        for phase in (0, 1):
            p = np.pad(thin, 1)
            n = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
            neighbours = sum(n)
            transitions = sum(((n[i] == 0) & (n[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
            a = n[0] * n[2] * (n[4] if phase == 0 else n[6])
            b = (n[2] if phase == 0 else n[0]) * n[4] * n[6]
            remove = (thin != 0) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & (a == 0) & (b == 0)
            if np.any(remove):
                changed = True
                thin[remove] = 0
        if not changed:
            break
    counts = cv2.filter2D(thin, cv2.CV_16S, np.ones((3, 3), np.int16), borderType=cv2.BORDER_CONSTANT) - thin
    yy, xx = np.where((thin != 0) & (counts <= 1))
    return np.column_stack((xx, yy)).astype(np.float32)


def _find_connections(rgb, text_mask, regions, background, sx, sy):
    geometric = [r for r in regions if r["shape"] != "text_group"]
    if len(geometric) < 2:
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    foreground = (gray < 130 if float(np.mean(background)) > 130 else gray > 155).astype(np.uint8) * 255
    foreground[text_mask != 0] = 0
    boxes = {}
    parent_ids = {region["parent_id"] for region in geometric if region["parent_id"]}
    for region in geometric:
        l, t, r, b = region["bbox"]
        box = [round(l / sx), round(t / sy), round(r / sx), round(b / sy)]
        boxes[region["id"]] = box
        if region["id"] in parent_ids:
            cv2.rectangle(foreground, (box[0], box[1]), (box[2], box[3]), 0, 7)
        else:
            cv2.rectangle(foreground, (max(0, box[0] - 3), max(0, box[1] - 3)), (box[2] + 3, box[3] + 3), 0, -1)
    foreground[:2] = foreground[-2:] = 0
    foreground[:, :2] = foreground[:, -2:] = 0
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    results = []
    gap = max(10, min(28, math.hypot(*gray.shape) * 0.025))
    for component in range(1, count):
        x, y, w, h, area = stats[component]
        if area < 10 or max(w, h) < 18 or area > gray.size * 0.08:
            continue
        local = (labels[y:y + h, x:x + w] == component).astype(np.uint8)
        yy, xx = np.where(local)
        points = np.column_stack((xx + x, yy + y)).astype(np.float32)
        endpoints = _skeleton_endpoints(local) + [x, y]
        if len(endpoints) < 2:
            continue
        attached = []
        for region_id, box in boxes.items():
            distance, end_index = _distance_to_box(endpoints, box)
            if distance <= gap:
                # Choose boundary evidence near an actual terminal, not a bend
                # that happens to pass close to an unrelated container.
                nearby = np.linalg.norm(points - endpoints[end_index], axis=1) < 16
                nearby_indices = np.where(nearby)[0]
                _, local_index = _distance_to_box(points[nearby], box)
                index = int(nearby_indices[local_index])
                attached.append((distance, region_id, index))
        if len(attached) != 2:
            continue
        attached.sort(key=lambda a: a[1])
        (_, source, start_index), (_, target, end_index) = attached
        start, end = points[start_index], points[end_index]
        span = float(np.linalg.norm(end - start))
        if span < 14:
            continue
        source_center = np.array([(boxes[source][0] + boxes[source][2]) / 2, (boxes[source][1] + boxes[source][3]) / 2])
        target_center = np.array([(boxes[target][0] + boxes[target][2]) / 2, (boxes[target][1] + boxes[target][3]) / 2])
        source_arrow = _arrow_at(points[np.linalg.norm(points - start, axis=1) < 48], source_center - start, span)
        target_arrow = _arrow_at(points[np.linalg.norm(points - end, axis=1) < 48], target_center - end, span)
        direction = "bidirectional" if source_arrow and target_arrow else "source_to_target" if target_arrow else "target_to_source" if source_arrow else "unknown"
        path = _path_in_component(local, start - [x, y], end - [x, y])
        results.append({"id": f"c{len(results) + 1}", "source": source, "target": target, "direction": direction, "kind": "connector_candidate", "points": [[round((px + x) * sx), round((py + y) * sy)] for px, py in path], "confidence": 0.7 if direction != "unknown" else 0.45, "confidence_basis": "uncalibrated geometric heuristic; not a probability", "label_text_ids": []})
        if len(results) >= 60:
            break
    return results


def _contrast(a, b):
    def luminance(rgb):
        s = np.array(rgb, dtype=float) / 255
        linear = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
        return float(linear @ [0.2126, 0.7152, 0.0722])
    l1, l2 = sorted([luminance(a), luminance(b)])
    return round((l2 + 0.05) / (l1 + 0.05), 2)


def _emphasis(rgb, text_regions, sx, sy):
    groups = {}
    height, width = rgb.shape[:2]
    for text in text_regions[:1500]:
        l, t, r, b = text["bbox"]
        l, t, r, b = max(0, round(l / sx)), max(0, round(t / sy)), min(width, round(r / sx)), min(height, round(b / sy))
        crop = rgb[t:b, l:r]
        if crop.size < 6:
            continue
        ring = rgb[max(0, t - 3):min(height, b + 3), max(0, l - 3):min(width, r + 3)]
        bg = _dominant(ring)
        pixels = crop.reshape(-1, 3)
        distances = np.linalg.norm(pixels.astype(float) - bg, axis=1)
        selected = pixels[distances >= max(30, float(np.percentile(distances, 80)))]
        if not len(selected):
            continue
        fg = _dominant(selected)
        foreground, background = _color(fg), _color(bg)
        key = (foreground["name"], background["name"])
        if key not in groups:
            groups[key] = {"text_ids": [], "foreground": foreground, "background": background, "contrast_ratio": _contrast(fg, bg)}
        groups[key]["text_ids"].append(text["id"])
    return sorted(groups.values(), key=lambda g: len(g["text_ids"]), reverse=True)[:12]


def _code_layout(text_regions):
    lines = []
    for text in sorted(text_regions, key=lambda t: ((t["bbox"][1] + t["bbox"][3]) / 2, t["bbox"][0])):
        l, top, r, bottom = text["bbox"]
        center = (top + bottom) / 2
        line = next((line for line in reversed(lines[-6:]) if abs(line["center"] - center) <= min(line["height"], bottom - top) * 0.45), None)
        if line is None:
            line = {"center": center, "height": bottom - top, "words": []}
            lines.append(line)
        line["words"].append(text)
    lines.sort(key=lambda line: line["center"])
    if not lines:
        return {"lines": [], "indent_step_px": None, "tabs_or_spaces": "unknown"}
    lefts = [min(word["bbox"][0] for word in line["words"]) for line in lines]
    char_widths = [(t["bbox"][2] - t["bbox"][0]) / max(1, len(t["text"])) for t in text_regions]
    char_width = float(np.median(char_widths)) if char_widths else 8
    base = min(lefts)
    offsets = sorted(set(round(left - base) for left in lefts))
    steps = [b - a for a, b in zip(offsets, offsets[1:]) if b - a > char_width * 1.5]
    step = float(min(steps)) if steps else None
    output = []
    for index, (line, left) in enumerate(zip(lines, lefts), 1):
        words = sorted(line["words"], key=lambda word: word["bbox"][0])
        boxes = [word["bbox"] for word in words]
        indent = round(left - base)
        output.append({"id": f"line{index}", "text_ids": [w["id"] for w in words], "bbox": [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)], "indent_px": indent, "indent_level": round(indent / step) if step else 0})
    return {"lines": output, "indent_step_px": step, "tabs_or_spaces": "unknown", "indentation_is_estimate": True}


def _graphic_candidates(rgb, text_mask, regions, sx, sy):
    output = []
    for region in regions:
        if region["shape"] == "text_group" or not region["fill"]:
            continue
        l, t, r, b = [round(value / scale) for value, scale in zip(region["bbox"], (sx, sy, sx, sy))]
        margin = max(4, min(r - l, b - t) // 15)
        l, t, r, b = l + margin, t + margin, r - margin, b - margin
        crop = rgb[t:b, l:r]
        if crop.size == 0:
            continue
        fill = np.array([int(region["fill"]["hex"][i:i + 2], 16) for i in (1, 3, 5)])
        mask = (np.linalg.norm(crop.astype(float) - fill, axis=2) > 65).astype(np.uint8) * 255
        mask[text_mask[t:b, l:r] != 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if w < 9 or h < 9 or area < 60 or w * h > crop.shape[0] * crop.shape[1] * 0.4:
                continue
            output.append({"id": f"g{len(output) + 1}", "kind": "graphic_candidate", "region_id": region["id"], "bbox": [round((l + x) * sx), round((t + y) * sy), round((l + x + w) * sx), round((t + y + h) * sy)], "meaning": None})
            if len(output) >= 80:
                return output
    return output


def _summary(context):
    regions, connections = context["regions"], context["connections"]
    parts = [f"The image is {context['image']['width']} × {context['image']['height']} pixels with a predominantly {context['background']['name']} outer background."]
    geometric = [r for r in regions if r["shape"] != "text_group"]
    if geometric:
        root = [r for r in geometric if r["parent_id"] is None]
        parts.append(f"Detected {len(geometric)} candidate containers, including {len(root)} outer containers; positions and contained OCR text are recorded by region ID.")
        for region in geometric[:10]:
            fill = region["fill"]["name"] if region["fill"] else "unmeasured"
            parent = f", inside {region['parent_id']}" if region['parent_id'] else ""
            parts.append(f"{region['id']}: {region['shape'].replace('_', ' ')}, row {region['row']}, column {region['column']}{parent}, {fill} fill, {len(region['text_ids'])} OCR word(s).")
    elif regions:
        parts.append(f"Detected {len(regions)} OCR text group(s); no enclosing geometric containers were confidently identified.")
    else:
        parts.append("No enclosing containers or OCR text groups were confidently identified.")
    if context["mode"] == "workflow":
        if connections:
            parts.append(f"Detected {len(connections)} connector candidate(s).")
            for c in connections[:15]:
                a, b, direction = c["source"], c["target"], c["direction"]
                relation = f"{a} → {b}" if direction == "source_to_target" else f"{b} → {a}" if direction == "target_to_source" else f"{a} ↔ {b}" if direction == "bidirectional" else f"{a} — {b} (direction unclear)"
                parts.append(relation + ".")
        else:
            parts.append("No connector candidates could be confidently linked to two detected containers; this does not establish that the image has no arrows.")
    else:
        parts.append(f"Grouped OCR words into {len(context['code_layout']['lines'])} visual line(s), with relative indentation recorded in pixels. Actual tabs and spaces cannot be recovered from pixel positions alone.")
    if context["graphics"]:
        parts.append(f"Detected {len(context['graphics'])} non-text graphic candidate(s) inside containers; identities are not inferred.")
    if context["emphasis"]:
        parts.append(f"Observed {len(context['emphasis'])} text colour/background combination(s); estimated contrast and text references are included.")
    parts.append("Colours describe appearance; no success, error, or workflow meaning is assigned without supporting evidence.")
    return "\n\n".join(parts)


def analyze_visual_context(image, text_regions, mode="workflow"):
    if mode not in {"workflow", "code"}:
        raise ValueError("Unknown visual context mode")
    original_width, original_height = image.size
    small = image.copy().convert("RGB")
    small.thumbnail((MAX_ANALYSIS_SIDE, MAX_ANALYSIS_SIDE), Image.Resampling.LANCZOS)
    rgb = np.asarray(small)
    sx, sy = original_width / small.width, original_height / small.height
    text_mask = _text_mask(rgb.shape, text_regions, sx, sy)
    edge = max(1, min(small.size) // 40)
    border_mask = np.zeros(rgb.shape[:2], np.uint8)
    border_mask[:edge] = border_mask[-edge:] = 1
    border_mask[:, :edge] = border_mask[:, -edge:] = 1
    border = rgb[(border_mask != 0) & (text_mask == 0)]
    background = _dominant(border if len(border) else rgb)
    regions = _find_regions(rgb, text_mask, background, text_regions, sx, sy)
    connections = _find_connections(rgb, text_mask, regions, background, sx, sy) if mode == "workflow" else []
    uncertainties = ["Local visual analysis uses geometric heuristics; regions and connectors should be checked against the preview.", "Colour meaning and icon or logo identities are not inferred. Contrast values are sampled estimates."]
    if any(c["direction"] == "unknown" for c in connections):
        uncertainties.append("Some connectors have no clear arrowhead evidence; their direction is unknown.")
    if mode == "workflow":
        uncertainties.append("Curved, dashed, crossing, low-contrast, or text-overlapping connectors can be missed. Connector labels are not automatically assigned.")
    else:
        uncertainties.append("Indentation is estimated from OCR positions; editor gutters and multiple panels can affect it. Code is not executed or semantically validated.")
    if len(text_regions) >= MAX_TEXT_REGIONS:
        uncertainties.append(f"Text layout is limited to the first {MAX_TEXT_REGIONS} detected OCR words; plain extracted text is retained separately.")
    if len(regions) >= MAX_REGIONS:
        uncertainties.append(f"Visual containers are limited to {MAX_REGIONS} candidates.")
    context = {"version": 1, "method": "local_geometry_v1", "mode": mode, "image": {"width": original_width, "height": original_height}, "coordinate_system": "pixels, origin top-left, bbox=[left,top,right,bottom]; EXIF-normalized image", "background": _color(background), "palette": _palette(rgb, text_mask), "regions": regions, "connections": connections, "text_regions": text_regions, "emphasis": _emphasis(rgb, text_regions, sx, sy), "graphics": _graphic_candidates(rgb, text_mask, regions, sx, sy), "code_layout": _code_layout(text_regions) if mode == "code" else None, "uncertainties": uncertainties}
    context["summary"] = _summary(context)
    return context
