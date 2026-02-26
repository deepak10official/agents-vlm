import base64
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from PIL import Image

load_dotenv()

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")