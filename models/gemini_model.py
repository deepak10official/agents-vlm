import base64
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
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


def main():
    image_path = Path(__file__).parent.parent / "documents" / "sample hdfc.jpg"
    image_b64 = image_file_to_base64(image_path)

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
