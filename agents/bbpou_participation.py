from datetime import date
from pathlib import Path
import sys
from typing import Any, Literal, Optional, Union
from langchain.agents import create_agent
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langfuse.langchain import CallbackHandler
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


agent = create_agent(
    model=model,            # add tools if needed
    response_format=BBPouParticipation
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