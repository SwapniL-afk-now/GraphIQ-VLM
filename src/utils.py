import string
import re
from PIL import Image
import os # Keep os import if resize_and_pad_image uses it implicitly, otherwise remove

from .config import VALID_DISEASES # Use relative import

def normalize_text(text):
    """Normalizes text by removing punctuation and lowercasing."""
    if not text: return ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = " ".join(text.split())
    return text

def extract_xml_tag(text: str, tag: str) -> str:
    """Extracts content within the specified XML tag."""
    if not text: return ""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return ""

def extract_disease(output_text: str) -> list | None:
    """
    Extracts the disease name from the <answer> tag and validates it
    against the VALID_DISEASES list from config.
    """
    if not output_text: return None
    answer_text = extract_xml_tag(output_text, "answer")

    if not answer_text:
        normalized_full_text = normalize_text(output_text)
        # Use VALID_DISEASES from config
        found_diseases = [disease for disease in VALID_DISEASES if disease in normalized_full_text]
        return found_diseases if found_diseases else None

    normalized_answer = normalize_text(answer_text)

    # Use VALID_DISEASES from config
    for disease in VALID_DISEASES:
        if normalized_answer == disease:
            return [disease]

    # Use VALID_DISEASES from config
    found_diseases = [disease for disease in VALID_DISEASES if disease in normalized_answer]
    return found_diseases if found_diseases else None

def resize_and_pad_image(image_path, size=(336, 336), fill_color=(0, 0, 0)):
    """
    Resizes an image maintaining aspect ratio and pads it to the target size.
    Returns a PIL Image object or None if an error occurs.
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    original_width, original_height = img.size
    target_width, target_height = size

    # Calculate resize ratio
    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize
    try:
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return None # Handle potential resize errors

    # Create padded background
    new_image = Image.new('RGB', size, fill_color)

    # Calculate paste position
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Paste image
    new_image.paste(img, (paste_x, paste_y))

    return new_image