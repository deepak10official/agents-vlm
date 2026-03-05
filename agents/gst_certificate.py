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
from utils.pdf_to_image import pdf_to_base64_image_parts

Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)

langfuse = get_client()
langfuse_handler = CallbackHandler()


class GstCertificateDetails(BaseModel):
    digital_signature: Literal["Yes", "No"] = Field(
        description='Whether a valid digital signature is done on the certificate. Use "No" if there is a question mark, "signature not verified", or similar indication; use "Yes" only when the digital signature is verified/done. Otherwise "No".'
    )
    physical_signature: Literal["Yes", "No"] = Field(
        description='Whether a physical (handwritten/inked) signature is present. Allowed values: "Yes" or "No".'
    )
    registration_number: str = Field(
        description="GST registration number (GSTIN) as stated in the certificate."
    )
    legal_name: str = Field(
        description="Legal name of the business/entity as mentioned in the certificate."
    )
    constitution_of_business: str = Field(
        description="Constitution of business (e.g. Private Limited, Partnership, Proprietorship) as stated in the certificate."
    )
    address_of_principal_place_of_business: str = Field(
        description="Full address of the principal place of business as stated in the certificate."
    )
    date_of_issue_of_certificate: Union[
        date,
        Literal["date is not mentioned"],
    ] = Field(
        default="date is not mentioned",
        description='Date of issue of the certificate in YYYY-MM-DD format, or the exact text "date is not mentioned".',
    )
    stamp_present: Literal["present", "not"] = Field(
        description='Only applies when physical_signature is Yes: whether an official stamp/seal is present alongside the physical signature. If only digital signature is done (no physical signature), use "not". Allowed values: "present" or "not".'
    )
    stamp_description: Optional[str] = Field(
        default=None,
        description="When stamp_present is 'present', the text or description visible on the stamp/seal (e.g. authority name, office). If no stamp or text not readable, use null.",
    )


GST_CERTIFICATE_SYSTEM_PROMPT = """You are an expert document analyst specializing in GST (Goods and Services Tax) certificates and registration documents.

Your task is to extract structured information from the provided document image(s) and return it in the exact schema required.

## Document context
- The document is typically a GST Registration Certificate or similar GST-related certificate issued by the tax authority.
- It may contain digital or physical signatures, stamps, registration number (GSTIN), legal name, constitution of business, and principal place of business address.

## Extraction rules

1. **digital_signature**: Use "Yes" only when a digital signature is done and verified on the certificate. Use "No" if there is a question mark, "signature not verified", or any indication that the signature is invalid or unverified; otherwise "No".

2. **physical_signature**: Answer "Yes" if the document shows or states a physical (handwritten/inked) signature; otherwise "No".

3. **registration_number**: Extract the GST registration number (GSTIN) exactly as stated in the document. Use a single string.

4. **legal_name**: Extract the legal name of the business/entity exactly as stated in the certificate.

5. **constitution_of_business**: Extract the constitution of business (e.g. Private Limited, Partnership, Proprietorship, LLP) as stated in the certificate.

6. **address_of_principal_place_of_business**: Extract the full address of the principal place of business as stated in the certificate.

7. **date_of_issue_of_certificate**: Extract the date of issue of the certificate in YYYY-MM-DD format if clearly stated. If no date is mentioned, use the exact string "date is not mentioned".

8. **stamp_present**: This rule applies only when there is a physical signature. When physical_signature is "Yes", answer "present" if an official stamp or seal is visible alongside the physical signature; otherwise "not". When only a digital signature is done (no physical signature), use "not".

9. **stamp_description**: When stamp_present is "present", extract the text or description visible on the stamp/seal (e.g. authority name, office, designation). If no stamp is present or the text is not readable, use null.

## Output requirements
- Return only the extracted fields in the required schema. Do not add commentary.
- Use the exact literal values "Yes" or "No" for digital_signature and physical_signature.
- Use the exact literal values "present" or "not" for stamp_present. For stamp_description, use null when not applicable.
- If a field cannot be determined from the document, use the most appropriate default or value as per the schema.
"""


agent = create_agent(
    model=model,
    response_format=GstCertificateDetails,
    system_prompt=GST_CERTIFICATE_SYSTEM_PROMPT,
)


def validate_document(agent: Any, document_path: str | Path) -> GstCertificateDetails:
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
                                "Extract GST certificate details from this document "
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
                "agent_type": "gst_certificate",
            },
            "tags": ["vlm", "pdf-validation"],
        },
    )

    return result["structured_response"]


if __name__ == "__main__":
    default_pdf_path = Path(__file__).resolve().parent.parent / "documents" / "pdfs" / "GST Certificate_Cashfree Payments India Private Limited.pdf"
    pdf_path = sys.argv[1] if len(sys.argv) >= 2 else default_pdf_path

    response = validate_document(agent, pdf_path)
    print(response.model_dump_json(indent=2))
