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

class BBPouParticipation(BaseModel):
    company_name: str = Field(
        description="Legal name of the company/entity."
    )
    types_of_entities: Literal["Bank", "Non-Bank"] = Field(
        description="Type of entity."
    )
    type_of_bbpou: Literal[
        "Customer BBPOU",
        "Biller BBPOU",
        "Both Customer and Biller BBPOU",
    ] = Field(description="BBPOU participation type.")
    address: str = Field(
        description="Registered or official address of the entity."
    )
    phone_number: str = Field(
        description="Contact phone number; kept as string to preserve formatting."
    )
    stamped_seal: Literal["Yes", "No"] = Field(
        description='Whether a stamped seal is present. Allowed values: "Yes" or "No".'
    )
    seal_description: Optional[str] = Field(
        default=None,
        description='Seal details if stamped_seal is "Yes"; otherwise null.',
    )
    authorized_signatory: Literal["Yes", "No"] = Field(
        description='Whether an authorized signatory is present. Allowed values: "Yes" or "No".'
    )
    signatory_name: Optional[str] = Field(
        default=None,
        description='Authorized signatory name if authorized_signatory is "Yes"; otherwise null.',
    )
    signatory_designation: Optional[str] = Field(
        default=None,
        description='Authorized signatory designation if authorized_signatory is "Yes"; otherwise null.',
    )
    date_of_authorization: Union[
        date,
        Literal["date is not mentioned"],
    ] = Field(
        default="date is not mentioned",
        description='Authorization date or the exact text "date is not mentioned".',
    )


BBPOU_SYSTEM_PROMPT = """You are an expert document analyst specializing in BBPOU (Bharat Bill Payment Operating Unit) participation letters and similar regulatory or onboarding documents.

Your task is to extract structured information from the provided document image(s) and return it in the exact schema required.

## Document context
- The document is typically a BBPOU participation letter, authorization letter, or similar formal letter from a company to a bill payment operator.
- It may contain company letterhead, stamps, seals, and signatures.

## Extraction rules

1. **company_name**: Extract the full legal name of the company or entity as it appears in the document (e.g. on letterhead or in the body). Use the official registered name, not abbreviations or trading names unless that is all that is stated.

2. **types_of_entities**: Classify the entity as either "Bank" or "Non-Bank" based on the document content and context (e.g. RBI registration, type of business).

3. **type_of_bbpou**: Choose exactly one of:
   - "Customer BBPOU" — if the entity is participating as a customer (bill payer) only
   - "Biller BBPOU" — if the entity is participating as a biller only
   - "Both Customer and Biller BBPOU" — if the document states participation in both roles

4. **address**: Extract the complete registered or official address of the entity as stated in the document (full address in one string).

5. **phone_number**: Extract the contact phone number exactly as written (including country code, spaces, or dashes). Keep as string; do not normalize format.

6. **stamped_seal**: Answer "Yes" if the document shows a company stamp, seal, or official stamp; otherwise "No".

7. **seal_description**: If stamped_seal is "Yes", briefly describe the seal (e.g. "Company round seal", "Official stamp with company name"). If "No", leave as null.

8. **authorized_signatory**: Answer "Yes" if the document clearly indicates an authorized signatory or signatory block; otherwise "No".

9. **signatory_name**: If authorized_signatory is "Yes", extract the full name of the signatory as printed or signed. If "No", leave as null.

10. **signatory_designation**: If authorized_signatory is "Yes", extract the person's designation (e.g. "Director", "Authorized Signatory", "CEO"). If "No", leave as null.

11. **date_of_authorization**: Extract the date of authorization, signing, or letter date in YYYY-MM-DD format if clearly stated. If no date is mentioned anywhere, use the exact string "date is not mentioned".

## Output requirements
- Return only the extracted fields in the required schema. Do not add commentary.
- For optional fields, use null when the condition for filling them is not met (e.g. seal_description when stamped_seal is "No").
- Use the exact literal values specified for enum-like fields (e.g. "Bank", "Non-Bank", "Yes", "No").
- If a field cannot be determined from the document, infer only when the context strongly supports it; otherwise use the most appropriate default or null as per the schema.
"""


agent = create_agent(
    model=model,
    response_format=BBPouParticipation,
    system_prompt=BBPOU_SYSTEM_PROMPT,
)


def validate_document(agent: Any, document_path: str | Path) -> BBPouParticipation:
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
                            "Extract BBPOU participation details from this document "
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
            "agent_type": "bbpou_validation",
        },
        "tags": ["vlm", "pdf-validation"],
    },
)

    return result["structured_response"]


if __name__ == "__main__":
    default_pdf_path = (
        r"C:\test\Agents\agents-vlm\documents\pdfs\BBPOU participation Letter_Cashfree Payments India Private Limited.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) >= 2 else default_pdf_path

    response = validate_document(agent, pdf_path)
    print(response.model_dump_json(indent=2))