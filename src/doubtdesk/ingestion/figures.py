import hashlib
import re
from collections import Counter
from statistics import median

import pymupdf

REPEAT_MIN_PAGES = 3
REPEAT_PAGE_FRACTION = 0.3
GLYPH_AREA = 0.005
STATS_MIN_AREA = 0.01
SCAN_AREA = 0.85
SCAN_MAX_NATIVE_CHARS = 50
LARGE_AREA = 0.6
FLAT_STD = 10
FAINT_MEAN = 215
FAINT_DARK = 0.02
TEXT_OVER_RATIO = 0.5
FIGURE_DENSITY_RATIO = 0.35
REGION_MIN_AREA = 0.02
REGION_MIN_SIDE = 40
TABLE_THIN_RATIO = 0.8
TABLE_MIN_PATHS = 4
THIN = 2.0
GRID = 6.0
REGION_GROWTH = 8.0
VISUAL_AREA = 0.05
GLYPH_PAGE_MIN = 20
GLYPH_MEDIAN_FACTOR = 3
STATS_MAX_PIXELS = 40000
XREF_MIN_PIXELS = 4096
BACKGROUND_LAYER = re.compile(r"watermark|background|artifact|stamp", re.IGNORECASE)
CAPTION = re.compile(
    r"^\s*(fig(?:ure)?s?|table|scheme|chart|diagram|plate|graph)\.?\s*"
    r"([A-Z]?\d+(?:\s*[.\-–]\s*\d+)*[a-z]?)\b",
    re.IGNORECASE | re.MULTILINE,
)
_KIND = {"fig": "figure", "figs": "figure", "figure": "figure", "figures": "figure"}


def normalize_ref(kind: str, number: str) -> str:
    kind = _KIND.get(kind.casefold(), kind.casefold())
    number = re.sub(r"\s*[.\-–]\s*", ".", number.strip()).casefold()
    return f"{kind} {number}"


QUESTION_REF = re.compile(
    r"\b(fig(?:ure)?s?|table|scheme|chart|diagram|plate|graph)\.?\s*(?:no\.?\s*)?"
    r"([A-Z]?\d+(?:\s*[.\-–]\s*\d+)*[a-z]?)\b",
    re.IGNORECASE,
)
_FIGURE_LIKE = {"diagram", "graph", "chart", "plate", "scheme"}


def question_refs(question: str) -> list[str]:
    refs = []
    for kind, number in QUESTION_REF.findall(question or ""):
        ref = normalize_ref(kind, number)
        refs.append(ref)
        if ref.split(" ", 1)[0] in _FIGURE_LIKE:
            refs.append("figure " + ref.split(" ", 1)[1])
    return list(dict.fromkeys(refs))


def caption_refs(text: str) -> list[str]:
    return list(dict.fromkeys(normalize_ref(kind, number) for kind, number in CAPTION.findall(text or "")))


def _rect_key(rect, step: float = 12.0):
    return tuple(round(value / step) for value in rect)


def _image_stats(pdf, xref):
    try:
        pix = pymupdf.Pixmap(pdf, xref)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        if pix.colorspace is None or pix.colorspace.n != 1:
            pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
        while pix.width * pix.height > STATS_MAX_PIXELS and min(pix.width, pix.height) > 4:
            pix.shrink(1)
        data = pix.samples
    except Exception:
        return None
    if not data:
        return None
    count = len(data)
    mean = sum(data) / count
    dark = sum(1 for value in data if value < 128) / count
    std = (sum((value - mean) ** 2 for value in data) / count) ** 0.5
    digest = hashlib.blake2b(bytes(data) + f"{pix.width}x{pix.height}".encode(), digest_size=16).hexdigest()
    return {"mean": mean, "dark": dark, "std": std, "digest": digest}


def layer_names(pdf) -> dict:
    try:
        return {xref: str(info.get("name", "")) for xref, info in (pdf.get_ocgs() or {}).items()}
    except Exception:
        return {}


def _in_background_layer(pdf, xref, layers: dict) -> bool:
    if not xref or not layers:
        return False
    try:
        kind, value = pdf.xref_get_key(xref, "OC")
    except Exception:
        return False
    if kind != "xref":
        return False
    match = re.match(r"(\d+)", value)
    return bool(match and BACKGROUND_LAYER.search(layers.get(int(match.group(1)), "")))


def page_features(pdf, page, layers: dict | None = None, stats_cache: dict | None = None) -> dict:
    layers = layer_names(pdf) if layers is None else layers
    stats_cache = {} if stats_cache is None else stats_cache
    rect = page.rect
    area = rect.get_area() or 1.0
    words = [(word[0], word[1], word[2], word[3], len(word[4])) for word in page.get_text("words")]
    blocks = [block[:4] for block in page.get_text("blocks") if block[6] == 0]
    placed = []
    for item in page.get_images(full=True):
        xref, smask, width, height = item[0], item[1], item[2], item[3]
        if width * height < XREF_MIN_PIXELS:
            continue
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        placed.extend((tuple(pymupdf.Rect(found) & rect), xref, bool(smask)) for found in rects)
    images = []
    for info in page.get_image_info():
        box = pymupdf.Rect(info["bbox"]) & rect
        if box.is_empty:
            continue
        fraction = box.get_area() / area
        xref, transparent = 0, False
        if fraction >= STATS_MIN_AREA:
            match = next((entry for entry in placed if all(abs(a - b) <= 1.5 for a, b in zip(entry[0], tuple(box)))), None)
            if match is not None:
                _, xref, transparent = match
        stats = None
        if xref and fraction >= STATS_MIN_AREA:
            if xref not in stats_cache:
                stats_cache[xref] = _image_stats(pdf, xref)
            stats = stats_cache[xref]
        images.append({
            "bbox": tuple(box), "area": fraction, "digest": stats["digest"] if stats else None,
            "position": _rect_key(tuple(box)), "transparent": transparent,
            "layer": _in_background_layer(pdf, xref, layers), "stats": stats,
        })
    paths = []
    for drawing in page.get_drawings():
        box = pymupdf.Rect(drawing["rect"]) & rect
        if box.is_empty and not (box.width or box.height):
            continue
        paths.append(tuple(box))
    return {
        "width": rect.width, "height": rect.height, "area": area,
        "words": words, "blocks": blocks, "images": images, "paths": paths,
        "native_chars": sum(word[4] for word in words),
    }


def _chars_in(words, box) -> int:
    x0, y0, x1, y1 = box
    return sum(count for wx0, wy0, wx1, wy1, count in words
               if x0 <= (wx0 + wx1) / 2 <= x1 and y0 <= (wy0 + wy1) / 2 <= y1)


def _box_area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _body_density(pages: list[dict]) -> float:
    values = []
    for page in pages:
        block_area = sum(_box_area(block) for block in page["blocks"])
        if block_area > 0 and page["native_chars"] > 0:
            values.append(page["native_chars"] / block_area)
    return median(values) if values else 0.0


def _cells(box, grow: float = 0.0):
    x0, y0, x1, y1 = box
    for gx in range(int((x0 - grow) // GRID), int((x1 + grow) // GRID) + 1):
        for gy in range(int((y0 - grow) // GRID), int((y1 + grow) // GRID) + 1):
            yield gx, gy


def _regions(boxes: list[tuple]) -> list[dict]:
    owner: dict = {}
    groups: list[list[int]] = []
    parent: list[int] = []

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, box in enumerate(boxes):
        parent.append(index)
        for cell in _cells(box, REGION_GROWTH):
            other = owner.get(cell)
            if other is None:
                owner[cell] = index
            else:
                a, b = find(index), find(other)
                if a != b:
                    parent[a] = b
    grouped: dict[int, list[int]] = {}
    for index in range(len(boxes)):
        grouped.setdefault(find(index), []).append(index)
    for members in grouped.values():
        groups.append(members)
    regions = []
    for members in groups:
        x0 = min(boxes[i][0] for i in members)
        y0 = min(boxes[i][1] for i in members)
        x1 = max(boxes[i][2] for i in members)
        y1 = max(boxes[i][3] for i in members)
        thin = sum(1 for i in members if boxes[i][2] - boxes[i][0] < THIN or boxes[i][3] - boxes[i][1] < THIN)
        regions.append({"bbox": (x0, y0, x1, y1), "paths": len(members), "thin": thin})
    return regions


def _union_fraction(boxes: list[tuple], area: float) -> float:
    cells = set()
    for box in boxes:
        cells.update(_cells(box))
    return min(1.0, len(cells) * GRID * GRID / area) if area else 0.0


def classify_image(image: dict, page: dict, repeated_digests: set, repeated_positions: set, body_density: float) -> str:
    if image["area"] < GLYPH_AREA:
        return "glyph"
    stats = image["stats"] or {}
    flat = bool(stats) and stats["std"] <= FLAT_STD
    faint = bool(stats) and stats["mean"] >= FAINT_MEAN and stats["dark"] <= FAINT_DARK
    strong = (
        image["layer"]
        or (image["digest"] is not None and image["digest"] in repeated_digests)
        or (image["position"] in repeated_positions and (image["transparent"] or flat or faint))
    )
    if strong:
        return "background"
    area_points = image["area"] * page["area"]
    text_density = _chars_in(page["words"], image["bbox"]) / area_points if area_points else 0.0
    text_over = body_density > 0 and text_density >= TEXT_OVER_RATIO * body_density
    weak = sum([image["transparent"], flat, faint, text_over, image["area"] >= LARGE_AREA])
    if weak >= 2:
        return "background"
    if image["area"] >= SCAN_AREA and page["native_chars"] < SCAN_MAX_NATIVE_CHARS:
        return "scan"
    return "figure"


def analyse_document(pages: list[dict]) -> list[dict]:
    count = len(pages)
    threshold = max(REPEAT_MIN_PAGES, REPEAT_PAGE_FRACTION * count)
    digests, positions, path_keys = Counter(), Counter(), Counter()
    for page in pages:
        digests.update({image["digest"] for image in page["images"] if image["digest"] is not None})
        positions.update({image["position"] for image in page["images"]})
        path_keys.update({_rect_key(path) for path in page["paths"]})
    repeated_digests = {key for key, value in digests.items() if value >= threshold}
    repeated_positions = {key for key, value in positions.items() if value >= threshold}
    template = {key for key, value in path_keys.items() if value >= threshold}
    body_density = _body_density(pages)

    results = []
    glyph_counts = []
    for page in pages:
        kinds = [classify_image(image, page, repeated_digests, repeated_positions, body_density) for image in page["images"]]
        figure_boxes = [image["bbox"] for image, kind in zip(page["images"], kinds) if kind == "figure"]
        content_paths = [path for path in page["paths"] if _rect_key(path) not in template]
        has_table = False
        for region in _regions(content_paths):
            box = region["bbox"]
            width, height = box[2] - box[0], box[3] - box[1]
            region_area = width * height
            if region["paths"] >= TABLE_MIN_PATHS and region["thin"] >= TABLE_THIN_RATIO * region["paths"]:
                has_table = has_table or region_area >= REGION_MIN_AREA * page["area"]
                continue
            if region_area < REGION_MIN_AREA * page["area"] or width < REGION_MIN_SIDE or height < REGION_MIN_SIDE:
                continue
            density = _chars_in(page["words"], box) / region_area
            if body_density <= 0 or density < FIGURE_DENSITY_RATIO * body_density:
                figure_boxes.append(box)
        glyphs = kinds.count("glyph")
        glyph_counts.append(glyphs)
        results.append({
            "figure_area": round(_union_fraction(figure_boxes, page["area"]), 4),
            "inline_image_count": glyphs,
            "has_table": has_table,
            "watermark_detected": "background" in kinds,
            "scanned": "scan" in kinds,
        })
    glyph_floor = max(GLYPH_PAGE_MIN, GLYPH_MEDIAN_FACTOR * median(glyph_counts)) if glyph_counts else GLYPH_PAGE_MIN
    for result in results:
        result["glyph_heavy"] = result["inline_image_count"] >= glyph_floor
    return results


def is_visual_page(figure_area: float, figure_refs: list[str], glyph_heavy: bool) -> bool:
    return (figure_area >= VISUAL_AREA
            or any(not ref.startswith("table ") for ref in figure_refs)
            or glyph_heavy)
