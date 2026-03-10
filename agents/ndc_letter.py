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


class NdcLetterDetails(BaseModel):
    bank_name: str = Field(
        description="Name of the bank issuing the letter."
    )
    bank_address: str = Field(
        description="Address of the bank as stated in the letter."
    )
    signature: Literal["Yes", "No"] = Field(
        description='Whether a signature is present on the letter. Allowed values: "Yes" or "No".'
    )
    designation_of_signer: str = Field(
        description="Designation or title of the person who signed the letter (e.g. Authorized Signatory, Manager)."
    )
    name_of_signer: str = Field(
        description="Name of the person who signed the letter."
    )
    net_cap_settlement_amount: str = Field(
        description="Net cap settlement amount as stated in the letter (e.g. currency and figure as written)."
    )
    net_cap_settlement_statement: Optional[str] = Field(
        default=None,
        description="The exact statement from the letter regarding net cap settlement amount (verbatim). If not found, use null.",
    )


NDC_LETTER_SYSTEM_PROMPT = """You are an expert document analyst specializing in NDC (Net Debit Cap) letters and sponsor bank letters.

Your task is to extract structured information from the provided document image(s) and return it in the exact schema required.

## Document context
- The document is typically an NDC letter or sponsor bank letter (e.g. sponsor letter PDF) containing bank details, signatory information, and net cap / settlement amount details.
- It may contain the bank's letterhead, address, signatory name and designation, and statements about net cap settlement amount.

## Extraction rules

1. **bank_name**: Extract the name of the bank issuing the letter.

2. **bank_address**: Extract the full address of the bank as stated in the letter.

3. **signature**: Answer "Yes" if a signature is visible or stated on the letter; otherwise "No".

4. **designation_of_signer**: Extract the designation or title of the person who signed (e.g. Authorized Signatory, Manager, Chief Manager). If no signatory is indicated, use "not stated" or empty string.

5. **name_of_signer**: Extract the name of the person who signed the letter. If not stated, use "not stated" or empty string.

6. **net_cap_settlement_amount**: Extract the net cap settlement amount as stated in the letter (e.g. the figure and currency exactly as written, e.g. "₹ X" or "INR X Lakhs").

7. **net_cap_settlement_statement**: Extract the exact statement from the letter that mentions the net cap settlement amount (verbatim, no paraphrasing). If no such statement is found, use null.

## Output requirements
- Return only the extracted fields in the required schema. Do not add commentary.
- Use the exact literal values "Yes" or "No" for signature.
- For net_cap_settlement_statement, use the exact wording from the document when present.
"""


agent = create_agent(
    model=model,
    response_format=NdcLetterDetails,
    system_prompt=NDC_LETTER_SYSTEM_PROMPT,
)


def validate_document(agent: Any, document_path: str | Path) -> NdcLetterDetails:
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
                                "Extract NDC letter details from this sponsor letter PDF "
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
                "agent_type": "ndc_letter",
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
        / "Sponsor Bank Letter_Cashfree Paymnents India Private Limited.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) >= 2 else default_pdf_path

    response = validate_document(agent, pdf_path)
    print(response.model_dump_json(indent=2))
