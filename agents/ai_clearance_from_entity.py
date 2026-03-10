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


class AiClearanceFromEntityDetails(BaseModel):
    not_agent_institution_confirmation: Literal["Yes", "No"] = Field(
        description='Whether the letter confirms that they are not an Agent Institution (AI) with any Operating Unit (OU) under the BBPS ecosystem. Use "Yes" if such a negative confirmation is clearly stated; otherwise "No".'
    )
    not_agent_institution_exact_sentences: Optional[str] = Field(
        default=None,
        description='When not_agent_institution_confirmation is "Yes", the exact sentence(s) from the letter that state they are not an AI with any OU under the BBPS ecosystem (verbatim). Otherwise null.',
    )
    sealed_or_stamped: Literal["Yes", "No"] = Field(
        description='Whether the letter has a seal or stamp. Allowed values: "Yes" or "No".'
    )
    seal_or_stamp_description: Optional[str] = Field(
        default=None,
        description="When sealed_or_stamped is 'Yes', the text or description visible on the seal/stamp (e.g. company name, designation). Otherwise null.",
    )
    company_name: str = Field(
        description="Name of the company/entity issuing the AI clearance letter as stated in the document."
    )


AI_CLEARANCE_FROM_ENTITY_SYSTEM_PROMPT = """You are an expert document analyst specializing in AI (Agent Institution) clearance letters and BBPS ecosystem related documents.

Your task is to extract structured information from the provided document image(s) and return it in the exact schema required.

## Document context
- The document is typically an AI clearance letter from an entity confirming that they are not an Agent Institution (AI) with any Operating Unit (OU) under the BBPS ecosystem.
- It may contain explicit negative confirmation wording, the company name, and seal/stamp.

## Extraction rules

1. **not_agent_institution_confirmation**: Answer "Yes" if the letter clearly confirms that the entity is NOT an Agent Institution (AI) with any Operating Unit (OU) under the BBPS ecosystem (e.g. wording like "we are not an AI with any OU under BBPS" or similar). Otherwise answer "No".

2. **not_agent_institution_exact_sentences**: When not_agent_institution_confirmation is "Yes", extract the exact sentence(s) in the letter that contain this negative confirmation (verbatim, without paraphrasing). When the confirmation is not present ("No"), use null.

3. **sealed_or_stamped**: Answer "Yes" if the document shows or mentions a seal or stamp; otherwise "No".

4. **seal_or_stamp_description**: When sealed_or_stamped is "Yes", extract the text or description visible on the seal or stamp (e.g. company name, branch, designation). When "No" or not readable, use null.

5. **company_name**: Extract the name of the company/entity issuing the letter exactly as stated in the document.

## Output requirements
- Return only the extracted fields in the required schema. Do not add commentary.
- Use the exact literal values "Yes" or "No" for not_agent_institution_confirmation and sealed_or_stamped.
- For not_agent_institution_exact_sentences, use the exact wording from the document when applicable.
"""


agent = create_agent(
    model=model,
    response_format=AiClearanceFromEntityDetails,
    system_prompt=AI_CLEARANCE_FROM_ENTITY_SYSTEM_PROMPT,
)


def validate_document(agent: Any, document_path: str | Path) -> AiClearanceFromEntityDetails:
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
                                "Extract AI clearance from entity details from this document "
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
                "agent_type": "ai_clearance_from_entity",
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
        / "AI clearance from entity_Cashfree Payments India Private Limited.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) >= 2 else default_pdf_path

    response = validate_document(agent, pdf_path)
    print(response.model_dump_json(indent=2))

