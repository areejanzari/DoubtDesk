import pymupdf

def count_characters(text: str) -> int:
    return len(text.strip())

def alphabetic_ratio(text: str) -> float:
    characters = [char for char in text if not char.isspace()]

    if not characters:
        return 0.0

    alphabetic_count = sum(char.isalpha() for char in characters)

    return alphabetic_count/len(characters)

def garbage_character_ratio(text: str) -> float:
    characters = [char for char in text if not char.isspace()]

    if not characters:
        return 0.0

    garbage_count = sum(char == "\uFFFD" for char in characters)

    return garbage_count/len(characters)


def image_coverage(page: pymupdf.Page) -> float:
    page_area = page.rect.get_area()

    if page_area == 0:
        return 0.0

    image_area = 0.0

    for image in page.get_image_info():
        image_rect = pymupdf.Rect(image["bbox"])
        visible_rect = image_rect & page.rect
        image_area += visible_rect.get_area()

    return min(image_area / page_area, 1.0)


def classify_page(
    text: str,
    coverage: float,
) -> str:
    char_count = count_characters(text)
    garbage_ratio = garbage_character_ratio(text)

    if char_count >= 100 and garbage_ratio > 0.1:
        return "ocr_candidate"

    if char_count >= 300:
        return "good_text"

    if char_count < 50 and coverage >= 0.8:
        return "ocr_candidate"

    return "low_text"

