from datetime import date
from pathlib import Path
import sys
from typing import Any, Literal, Optional, Union
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
import os


from models.gemini import model
#from models.groq import model
from utils.pdf_to_image import pdf_to_base64_image_parts


Langfuse(public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"))

langfuse = get_client()

# Initialize the Langfuse handler
langfuse_handler = CallbackHandler()

class CertificateOfAuthorizationDetails(BaseModel):
    rbi_approval_received: Literal["Yes", "No"] = Field(
        description='Whether the entity has received approval from the RBI. Allowed values: "Yes" or "No".'
    )
    approval_statement_exact: Optional[str] = Field(
        default=None,
        description='If rbi_approval_received is "Yes", the exact approval statement as written in the document; otherwise null.',
    )
    location: str = Field(
        description="Location mentioned in the certificate (e.g. city, office location, place of issue)."
    )
    date_of_issue: Union[
        date,
        Literal["date is not mentioned"],
    ] = Field(
        default="date is not mentioned",
        description='Date of issue of the certificate in YYYY-MM-DD format, or the exact text "date is not mentioned".',
    )
    signature_chief_general_manager: Literal["Yes", "No"] = Field(
        description='Whether the signature of the Chief General Manager-in-Charge is present. Allowed values: "Yes" or "No".'
    )
    certification_of_authorisation_number: str = Field(
        description="The certification of authorisation number as stated in the document."
    )


CERTIFICATE_OF_AUTHORIZATION_SYSTEM_PROMPT = """You are an expert document analyst specializing in Certificates of Authorization and RBI-related regulatory documents.

Your task is to extract structured information from the provided document image(s) and return it in the exact schema required.

## Document context
- The document is typically a Certificate of Authorization issued by or in connection with RBI (Reserve Bank of India) or similar regulatory authority.
- It may contain stamps, seals, signatures of the Chief General Manager-in-Charge, and certification details.

## Extraction rules

1. **rbi_approval_received**: Answer "Yes" if the document states or implies that the entity has received approval from the RBI; otherwise "No".

2. **approval_statement_exact**: If rbi_approval_received is "Yes", extract the exact approval statement as it appears in the document (verbatim, no paraphrasing). If "No", leave as null.

3. **location**: Extract the location mentioned in the certificate (e.g. city, office location, place of issue, "Mumbai", "Head Office", etc.). Use a single string.

4. **date_of_issue**: Extract the date of issue of the certificate in YYYY-MM-DD format if clearly stated. If no date is mentioned, use the exact string "date is not mentioned".

5. **signature_chief_general_manager**: Answer "Yes" if the document shows or states the signature of the Chief General Manager-in-Charge; otherwise "No".

6. **certification_of_authorisation_number**: Extract the certification of authorisation number exactly as stated in the document (e.g. alphanumeric reference, certificate number). Keep as string.

## Output requirements
- Return only the extracted fields in the required schema. Do not add commentary.
- For approval_statement_exact, use the exact wording from the document when rbi_approval_received is "Yes".
- Use the exact literal values "Yes" or "No" for boolean-like fields.
- If a field cannot be determined from the document, use the most appropriate default or null as per the schema.
"""


agent = create_agent(
    model=model,
    response_format=CertificateOfAuthorizationDetails,
    system_prompt=CERTIFICATE_OF_AUTHORIZATION_SYSTEM_PROMPT,
)


def validate_document(agent: Any, document_path: str | Path) -> CertificateOfAuthorizationDetails:
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
                            "Extract Certificate of Authorization details from this document "
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
            "agent_type": "certificate_of_authorization",
        },
        "tags": ["vlm", "pdf-validation"],
    },
)

    return result["structured_response"]


if __name__ == "__main__":
    default_pdf_path = (
        r"C:\test\Agents\agents-vlm\documents\pdfs\Certificate Of Authorization_Cashfree Payments India Private Limited.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) >= 2 else default_pdf_path

    response = validate_document(agent, pdf_path)
    print(response.model_dump_json(indent=2))
