import base64
from io import BytesIO
from pathlib import Path

from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from PIL import Image

model = ChatOllama(model="qwen3-vl:2b", temperature=0)