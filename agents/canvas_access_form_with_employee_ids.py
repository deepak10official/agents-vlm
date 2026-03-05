from pathlib import Path
import sys
from typing import Any, Literal
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
import os

from models.gemini import model
#from models.ollama import model
from utils.pdf_to_image import pdf_to_base64_image_parts

Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)
langfuse = get_client()
langfuse_handler = CallbackHandler()


class AdminDetails(BaseModel):
    """Details for a single admin (admin 1 or admin 2). Use 'VALUES MISSING' if not present in the document."""
    name: str = Field(default="VALUES MISSING", description="Full name of the admin. Use 'VALUES MISSING' if not present.")
    department: str = Field(default="VALUES MISSING", description="Department of the admin. Use 'VALUES MISSING' if not present.")
    designation: str = Field(default="VALUES MISSING", description="Designation/role of the admin. Use 'VALUES MISSING' if not present.")
    mobile_number: str = Field(default="VALUES MISSING", description="Mobile number of the admin. Use 'VALUES MISSING' if not present.")
    emailid: str = Field(default="VALUES MISSING", description="Email ID of the admin. Use 'VALUES MISSING' if not present.")
    access_type: str = Field(default="VALUES MISSING", description="Type of access granted to the admin. Use 'VALUES MISSING' if not present.")


class AuthorisedDetails(BaseModel):
    """Details when the signatory is authorised. Use 'VALUES MISSING' if not present or not applicable."""
    name: str = Field(default="VALUES MISSING", description="Name of the authorised person. Use 'VALUES MISSING' if not present.")
    designation: str = Field(default="VALUES MISSING", description="Designation of the authorised person. Use 'VALUES MISSING' if not present.")
    signature_present: Literal["yes", "VALUES MISSING"] = Field(
        default="VALUES MISSING",
        description="Whether signature is present (signed). Allowed values only: 'yes' or 'VALUES MISSING'.",
    )
    seal_and_stamp: Literal["yes", "VALUES MISSING"] = Field(
        default="VALUES MISSING",
        description="Whether seal and stamp are present. Allowed values only: 'yes' or 'VALUES MISSING'.",
    )
    seal_and_stamp_description: str = Field(
        default="VALUES MISSING",
        description="Description of the seal and stamp. Use 'VALUES MISSING' if not present.",
    )


class CanvasAccessFormWithEmployeeIds(BaseModel):
    name_of_the_bbpou: str = Field(default="VALUES MISSING", description="Name of the BBPOU. Use 'VALUES MISSING' if not present.")
    bbpou_id: str = Field(default="VALUES MISSING", description="BBPOU ID. Use 'VALUES MISSING' if not present.")
    date_of_request: str = Field(default="VALUES MISSING", description="Date of the request. Use 'VALUES MISSING' if not present.")
    admin_1: AdminDetails = Field(description="Details for Admin 1.")
    admin_2: AdminDetails = Field(description="Details for Admin 2.")
    declaration: Literal["present", "VALUES MISSING"] = Field(
        default="VALUES MISSING",
        description="Whether the declaration is present or not. Allowed values only: 'present' or 'VALUES MISSING'.",
    )
    declaration_description: str = Field(
        description="Whether the declaration text matches the expected text. Allowed values: 'matched perfectly' or 'not matched, it is edited'.",
    )
    is_authorised: Literal["authorised", "not authorised", "VALUES MISSING"] = Field(
        default="VALUES MISSING",
        description="Whether the document/signatory is authorised or not. Allowed values only: 'authorised' or 'not authorised' or 'VALUES MISSING'.",
    )
    authorised_details: AuthorisedDetails = Field(
        description="When is_authorised is 'authorised': name, designation, signature_present (yes or VALUES MISSING), seal_and_stamp (yes or VALUES MISSING), seal_and_stamp_description. Otherwise use VALUES MISSING for fields.",
    )


CANVAS_ACCESS_FORM_SYSTEM_PROMPT = """You are an expert document analyst specializing in Canvas Access Forms and BBPOU (Business Correspondent Point of Operation) documents.

Your task is to extract structured information from the provided document image(s) and return it in the exact schema required.

## Document context
- The document is a Canvas Access Form, typically containing BBPOU details, admin 1 and admin 2 details (name, department, designation, mobile number, email, access type), a declaration section, and an authorised signatory section with name, designation, signature, and seal/stamp.

## Extraction rules

1. **name_of_the_bbpou**: Extract the name of the BBPOU. Use "VALUES MISSING" if not present.

2. **bbpou_id**: Extract the BBPOU ID. Use "VALUES MISSING" if not present.

3. **date_of_request**: Extract the date of request. Use "VALUES MISSING" if not present.

4. **admin_1** and **admin_2**: For each admin, extract name, department, designation, mobile_number, emailid, access_type. Use "VALUES MISSING" for any field not present in the document.

5. **declaration**: Use "present" if a declaration section is present in the document; otherwise "VALUES MISSING".

6. **declaration_description**: Use "matched perfectly" if the declaration text matches the expected standard text; use "not matched, it is edited" if it appears edited or different. Required field.

7. **is_authorised**: Use "authorised" if the document/signatory is indicated as authorised; use "not authorised" otherwise. Use "VALUES MISSING" only if not determinable.

8. **authorised_details**: When is_authorised is "authorised", extract name, designation, signature_present ("yes" if signature is present, else "VALUES MISSING"), seal_and_stamp ("yes" if seal and stamp are present, else "VALUES MISSING"), seal_and_stamp_description. When not authorised or not applicable, use "VALUES MISSING" for each field.

## Output requirements
- Return only the extracted fields in the required schema. Do not add commentary.
- Use the exact string "VALUES MISSING" (all caps) for any missing or undeterminable value where allowed.
- For declaration_description always provide either "matched perfectly" or "not matched, it is edited".
"""


agent = create_agent(
    model=model,
    response_format=CanvasAccessFormWithEmployeeIds,
    system_prompt=CANVAS_ACCESS_FORM_SYSTEM_PROMPT,
)


def validate_document(agent: Any, document_path: str | Path) -> CanvasAccessFormWithEmployeeIds:
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
                                "Extract Canvas Access Form details from this document "
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
                "agent_type": "canvas_access_form_with_employee_ids",
            },
            "tags": ["vlm", "pdf-validation"],
        },
    )
    return result["structured_response"]


if __name__ == "__main__":
    default_pdf_path = (
        r"C:\test\Agents\agents-vlm\documents\pdfs\CH51_Canvas_Access Form.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) >= 2 else default_pdf_path
    response = validate_document(agent, pdf_path)
    print(response.model_dump_json(indent=2))
