from pathlib import Path

import pymupdf

PAGE = pymupdf.paper_rect("a4")
TOPICS = ["kidney", "nephron", "glomerulus", "tubule", "ureter", "bladder", "urea", "filtrate", "medulla", "cortex"]
VERBS = ["filters", "stores", "moves", "concentrates", "regulates", "carries", "absorbs", "secretes"]


def _pattern_pixmap(width=240, height=240, seed=0, alpha=False):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), alpha)
    pix.clear_with(255)
    step = 12 + seed % 7
    for index, x in enumerate(range(0, width, step)):
        shade = (index * 37 + seed * 53) % 200
        pix.set_rect(pymupdf.IRect(x, 0, x + step // 2, height), (shade, 40 + shade // 3, 255 - shade) + ((255,) if alpha else ()))
    for y in range(0, height, step * 2):
        pix.set_rect(pymupdf.IRect(0, y, width, y + 3), (0, 0, 0) + ((255,) if alpha else ()))
    return pix


def _flat_pixmap(value=0, alpha_value=40):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 400), True)
    pix.clear_with(value)
    pix.set_alpha(bytes([alpha_value]) * (300 * 400))
    return pix


def _body_text(page, number, top=110, lines=26):
    for line in range(lines):
        topic = TOPICS[(number * 3 + line) % len(TOPICS)]
        verb = VERBS[(number + line * 5) % len(VERBS)]
        other = TOPICS[(number * 7 + line * 3 + 1) % len(TOPICS)]
        text = f"The {topic} {verb} fluid towards the {other} during step {line} of section {number}."
        if number == 1 and line == 0:
            text = "The kidney filters blood and regulates water balance in the body."
        page.insert_text((60, top + line * 22), text, fontsize=10)


def _template(page, number):
    page.insert_text((60, 40), "BIOLOGY TEXTBOOK CHAPTER", fontsize=9)
    page.insert_text((280, 815), f"Page {number}", fontsize=9)
    page.draw_line((50, 55), (545, 55), color=(0.4, 0.4, 0.4), width=0.6)


def build(path: Path, kind: str, pages: int = 6) -> Path:
    with pymupdf.open() as pdf:
        watermark = _flat_pixmap()
        for number in range(1, pages + 1):
            page = pdf.new_page(width=PAGE.width, height=PAGE.height)
            if kind == "scan":
                page.insert_image(page.rect, pixmap=_pattern_pixmap(600, 840, seed=number))
                continue
            _template(page, number)
            _body_text(page, number, lines=10 if kind in {"vector_diagram", "captionless_figure"} else 26)
            if kind == "image_watermark":
                page.insert_image(pymupdf.Rect(100, 200, 500, 700), pixmap=watermark, overlay=True)
            elif kind == "text_watermark":
                page.insert_text((300, 600), "NOT TO BE REPUBLISHED", fontsize=40, rotate=90, color=(0.85, 0.85, 0.85))
            elif kind == "page_tint":
                page.draw_rect(page.rect, color=None, fill=(0.97, 0.96, 0.9), overlay=False)
            elif kind == "full_page_diagram" and number == 3:
                page.clean_contents()
                page.insert_image(pymupdf.Rect(40, 70, 555, 800), pixmap=_pattern_pixmap(500, 700, seed=3))
            elif kind == "vector_diagram" and number == 2:
                shape = page.new_shape()
                for index in range(12):
                    shape.draw_circle((300, 420), 20 + index * 12)
                for index in range(10):
                    shape.draw_line((120, 300 + index * 25), (480, 540 - index * 25))
                shape.finish(color=(0, 0, 0), width=1)
                shape.commit()
            elif kind == "captionless_figure" and number == 4:
                page.insert_image(pymupdf.Rect(150, 300, 450, 560), pixmap=_pattern_pixmap(seed=9))
            elif kind == "shadow_text" and number == 1:
                page.insert_text((60, 720), "ammonotelic", fontsize=10)
                page.insert_text((60.2, 720.1), "ammonotelic", fontsize=10)
        pdf.save(path)
    return path
