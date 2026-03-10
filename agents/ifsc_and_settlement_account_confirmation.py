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


class IfscAndSettlementAccountConfirmationDetails(BaseModel):
    bank_name: str = Field(
        description="Name of the bank as stated in the IFSC and settlement account confirmation document."
    )
    bank_ifsc_code: str = Field(
        description="IFSC code of the bank/branch as stated in the document."
    )
    bank_account_number: str = Field(
        description="Bank account number (settlement account number) as stated in the document."
    )
    signed: Literal["Yes", "No"] = Field(
        description='Whether the document is signed. Allowed values: "Yes" or "No".'
    )
    name_of_signer: str = Field(
        description="Name of the person who signed the document. If not stated, use 'not stated' or empty string."
    )
    designation: str = Field(
        description="Designation or title of the person who signed (e.g. Authorized Signatory, Manager)."
    )
    sealed_or_stamped: Literal["Yes", "No"] = Field(
        description='Whether the document has a seal or stamp. Allowed values: "Yes" or "No".'
    )
    stamp_seal_description: Optional[str] = Field(
        default=None,
        description="When sealed_or_stamped is 'Yes', the text or description visible on the stamp/seal. Otherwise null.",
    )


IFSC_AND_SETTLEMENT_ACCOUNT_CONFIRMATION_SYSTEM_PROMPT = """You are an expert document analyst specializing in IFSC and settlement account confirmation letters/documents.

Your task is to extract structured information from the provided document image(s) and return it in the exact schema required.

## Document context
- The document is typically an IFSC and Settlement Account Confirmation letter or similar confirmation from a bank, containing bank name, IFSC code, account number, and signatory/seal details.

## Extraction rules

1. **bank_name**: Extract the name of the bank as stated in the document.

2. **bank_ifsc_code**: Extract the IFSC code of the bank or branch exactly as stated in the document.

3. **bank_account_number**: Extract the bank account number (settlement account number) as stated in the document.

4. **signed**: Answer "Yes" if the document is signed (signature visible or stated); otherwise "No".

5. **name_of_signer**: Extract the name of the person who signed the document. If not stated, use "not stated" or empty string.

6. **designation**: Extract the designation or title of the person who signed (e.g. Authorized Signatory, Manager). If not stated, use "not stated" or empty string.

7. **sealed_or_stamped**: Answer "Yes" if the document has a seal or stamp visible or stated; otherwise "No".

8. **stamp_seal_description**: When sealed_or_stamped is "Yes", extract the text or description visible on the stamp or seal (e.g. authority name, branch). When "No", use null.

## Output requirements
- Return only the extracted fields in the required schema. Do not add commentary.
- Use the exact literal values "Yes" or "No" for signed and sealed_or_stamped.
- For stamp_seal_description, use null when sealed_or_stamped is "No".
"""


agent = create_agent(
    model=model,
    response_format=IfscAndSettlementAccountConfirmationDetails,
    system_prompt=IFSC_AND_SETTLEMENT_ACCOUNT_CONFIRMATION_SYSTEM_PROMPT,
)


def validate_document(agent: Any, document_path: str | Path) -> IfscAndSettlementAccountConfirmationDetails:
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
                                "Extract IFSC and settlement account confirmation details from this document "
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
                "agent_type": "ifsc_and_settlement_account_confirmation",
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
        / "IFSC and Settlement Ac Confirmation_Cashfree Payments India Private Limited.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) >= 2 else default_pdf_path

    response = validate_document(agent, pdf_path)
    print(response.model_dump_json(indent=2))
