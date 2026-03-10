from pathlib import Path
import sys
from typing import Any, Literal, Optional
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
import os

from models.gemini import model
from utils.pdf_to_image import pdf_to_base64_image_parts

Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)

langfuse = get_client()
langfuse_handler = CallbackHandler()


class CommencementLetterToRbiDetails(BaseModel):
    sender_name: str = Field(
        description='Name of the sender of the letter. If not present or not mentioned, use exactly "Value missing".'
    )
    sender_mail_id: str = Field(
        description='Sender email address as in the letter. If not present or not mentioned, use exactly "Value missing".'
    )
    receiver_mail_id: str = Field(
        description='Receiver email address as in the letter. If not present or not mentioned, use exactly "Value missing".'
    )
    cc_s: str = Field(
        description='CC (carbon copy) recipients - list of email addresses or "CC" field content. If not present or none, use exactly "Value missing".'
    )
    commencement_date_present: Literal["Yes", "No"] = Field(
        description='Whether a commencement date is mentioned in the letter. Allowed values: "Yes" or "No".'
    )
    commencement_date: Optional[str] = Field(
        default=None,
        description='When commencement_date_present is "Yes", the commencement date as stated (e.g. DD-MM-YYYY or as written). Otherwise null.',
    )
    commencement_exact_sentences: Optional[str] = Field(
        default=None,
        description='When commencement_date_present is "Yes", the exact sentence(s) from the letter that mention the commencement date (verbatim). Otherwise null.',
    )


COMMENCEMENT_LETTER_TO_RBI_SYSTEM_PROMPT = """You are an expert document analyst specializing in official letters to RBI (Reserve Bank of India), including commencement letters.

Your task is to extract structured information from the provided document image(s) and return it in the exact schema required.

## Document context
- The document is typically a Commencement letter to RBI or similar formal letter (e.g. from an entity to RBI).
- It may contain sender/receiver details, email IDs, CC list, and a commencement date with related sentences.

## Extraction rules

1. **sender_name**: Extract the name of the sender of the letter. If not present or not mentioned in the document, use exactly "Value missing".

2. **sender_mail_id**: Extract the sender's email address as shown in the letter. If not present or not mentioned, use exactly "Value missing".

3. **receiver_mail_id**: Extract the receiver's (e.g. RBI) email address as shown in the letter. If not present or not mentioned, use exactly "Value missing".

4. **cc_s**: Extract the CC (carbon copy) recipients - all email addresses or text in the CC field. If there is no CC or the field is not present, use exactly "Value missing".

5. **commencement_date_present**: Answer "Yes" if the letter explicitly mentions a commencement date (date of commencement); otherwise "No".

6. **commencement_date**: When commencement_date_present is "Yes", extract the commencement date as stated in the letter (preserve format, e.g. DD-MM-YYYY or as written). When "No", use null.

7. **commencement_exact_sentences**: When commencement_date_present is "Yes", extract the exact sentence(s) from the letter that mention the commencement date (verbatim, no paraphrasing). When "No", use null.

## Output requirements
- Return only the extracted fields in the required schema. Do not add commentary.
- For sender_name, sender_mail_id, receiver_mail_id, and cc_s: use exactly "Value missing" (with capital V and M) when the value is not present or not mentioned.
- Use the exact literal values "Yes" or "No" for commencement_date_present.
- For commencement_exact_sentences, use the exact wording from the document when commencement date is present.
"""


agent = create_agent(
    model=model,
    response_format=CommencementLetterToRbiDetails,
    system_prompt=COMMENCEMENT_LETTER_TO_RBI_SYSTEM_PROMPT,
)


def validate_document(agent: Any, document_path: str | Path) -> CommencementLetterToRbiDetails:
    image_parts = pdf_to_base64_image_parts(document_path)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract commencement letter to RBI details from this document "
                                "and return the structured response."
                            ),
                        },
                        *image_parts,
                    ],
                }
            ]
        },
        config={
            "callbacks": [langfuse_handler],
            "metadata": {
                "document_path": str(document_path),
                "agent_type": "commencement_letter_to_rbi",
            },
            "tags": ["vlm", "pdf-validation"],
        },
    )

    return result["structured_response"]


if __name__ == "__main__":
    default_pdf_path = (
        Path(__file__).resolve().parent.parent
        / "documents"
        / "pdfs"
        / "Commencement letter to RBI_Cashfree Paymnents India Private Limited.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) >= 2 else default_pdf_path

    response = validate_document(agent, pdf_path)
    print(response.model_dump_json(indent=2))
