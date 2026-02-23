import base64
from io import BytesIO
from pathlib import Path

from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from PIL import Image

llm = ChatOllama(model="qwen3-vl:2b", temperature=0)


def image_file_to_base64(image_path: Path) -> str:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        buf = BytesIO()
        rgb.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


def main():
    image_path = Path(__file__).with_name("sample_bank_letter.png")
    image_b64 = image_file_to_base64(image_path)

    chain = prompt_func | llm | StrOutputParser()

    query_chain = chain.invoke(
        {"text": "do we have any stamps there?", "image": image_b64}
    )

    print(query_chain)


if __name__ == "__main__":
    main()