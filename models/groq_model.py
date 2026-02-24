import base64
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from PIL import Image

load_dotenv()

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


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
            {"type": "text", "text": "i want to know wheather seal is used or not, if used then what is on the seal"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
    )

    response = model.invoke([message])
    print(response.content)


if __name__ == "__main__":
    main()
