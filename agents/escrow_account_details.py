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


class EscrowAccountDetails(BaseModel):
    account_name: str = Field(
        description="Name of the escrow account as stated in the document."
    )
    account_number: str = Field(
        description="Escrow account number as stated in the document."
    )
    account_opening_date: str = Field(
        description="Date of account opening as stated in the document (e.g. DD-MM-YYYY or as written). If not mentioned, use 'not mentioned'."
    )
    ifsc_code: str = Field(
        description="IFSC code of the bank/branch for the escrow account as stated in the document."
    )
    signed: Literal["Yes", "No"] = Field(
        description='Whether the document is signed. Allowed values: "Yes" or "No".'
    )
    signed_date: Optional[str] = Field(
        default=None,
        description="When signed is 'Yes', the date of signing as stated in the document. Otherwise null.",
    )
    seal_or_stamp: Literal["Yes", "No"] = Field(
        description='Whether the document has a seal or stamp. Allowed values: "Yes" or "No".'
    )
    seal_description: Optional[str] = Field(
        default=None,
        description="When seal_or_stamp is 'Yes', the text or description visible on the seal/stamp. Otherwise null.",
    )


ESCROW_ACCOUNT_DETAILS_SYSTEM_PROMPT = """You are an expert document analyst specializing in escrow account details and bank account documents.

Your task is to extract structured information from the provided document image(s) and return it in the exact schema required.

## Document context
- The document is typically an escrow account details letter or confirmation from a bank, containing account name, account number, opening date, IFSC code, and signatory/seal details.

## Extraction rules

1. **account_name**: Extract the name of the escrow account as stated in the document.

2. **account_number**: Extract the escrow account number exactly as stated in the document.

3. **account_opening_date**: Extract the account opening date as stated in the document (preserve format, e.g. DD-MM-YYYY). If not mentioned, use "not mentioned".

4. **ifsc_code**: Extract the IFSC code of the bank or branch for this escrow account as stated in the document.

5. **signed**: Answer "Yes" if the document is signed (signature visible or stated); otherwise "No".

6. **signed_date**: When signed is "Yes", extract the date of signing as stated in the document (if any). When signed is "No" or date not stated, use null.

7. **seal_or_stamp**: Answer "Yes" if the document has a seal or stamp visible or stated; otherwise "No".

8. **seal_description**: When seal_or_stamp is "Yes", extract the text or description visible on the seal or stamp (e.g. authority name, branch). When "No", use null.

## Output requirements
- Return only the extracted fields in the required schema. Do not add commentary.
- Use the exact literal values "Yes" or "No" for signed and seal_or_stamp.
- For signed_date and seal_description, use null when not applicable.
"""


agent = create_agent(
    model=model,
    response_format=EscrowAccountDetails,
    system_prompt=ESCROW_ACCOUNT_DETAILS_SYSTEM_PROMPT,
)


def validate_document(agent: Any, document_path: str | Path) -> EscrowAccountDetails:
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
                                "Extract escrow account details from this document "
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
                "agent_type": "escrow_account_details",
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
        / "Escrow account details_Cashfree Payments India Private Limited.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) >= 2 else default_pdf_path

    response = validate_document(agent, pdf_path)
    print(response.model_dump_json(indent=2))
