import base64
import tempfile
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import fitz  # PyMuPDF
from PIL import Image

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
)


def image_file_to_base64(image_path: Path) -> str:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        buf = BytesIO()
        rgb.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pdf_file_to_image_paths(pdf_path: Path, output_dir: Path, dpi: int = 300) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []

    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_path = output_dir / f"page_{page_num + 1}.jpg"
        img.save(image_path, "JPEG")
        image_paths.append(image_path)

    return image_paths


def main():
    input_path = (
        Path(__file__).parent.parent
        / "documents"
        / "pdfs"
        / "Letter of authority_Cashfree Paymnents India Private Limited.pdf"
    )

    if input_path.suffix.lower() == ".pdf":
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_paths = pdf_file_to_image_paths(input_path, Path(tmp_dir))
            image_parts = [
                {
                    "type": "image",
                    "base64": image_file_to_base64(image_path),
                    "mime_type": "image/jpeg",
                }
                for image_path in image_paths
            ]
            message = HumanMessage(
                content=[{"type": "text", "text": "wheather seal is used or not, if used then what is on the seal"}] + image_parts
            )
            response = model.invoke([message])
            print(response.content)
    else:
        image_b64 = image_file_to_base64(input_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in detail."},
                {
                    "type": "image",
                    "base64": image_b64,
                    "mime_type": "image/jpeg",
                },
            ]
        )
        response = model.invoke([message])
        print(response.content)


if __name__ == "__main__":
    main()
