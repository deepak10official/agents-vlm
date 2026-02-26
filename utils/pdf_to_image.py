import base64
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to base64 encoded JPEG."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pdf_to_base64_image_parts(pdf_path: str | Path, dpi: int = 300) -> list[dict[str, str]]:
    """Convert all pages of a PDF into LangChain-compatible base64 image parts."""
    input_path = Path(pdf_path)
    if not input_path.exists():
        raise FileNotFoundError(f"PDF file not found: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {input_path.suffix}")

    image_parts: list[dict[str, str]] = []
    with fitz.open(input_path) as document:
        for page in document:
            pixmap = page.get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            image_parts.append(
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": image_to_base64(image),
                    "mime_type": "image/jpeg",
                }
            )

    return image_parts