
from typing import List, Optional
from pydantic import BaseModel, Field


class StructuredFormat(BaseModel):
    """A dynamic test-result section with a title and an informative description."""
    title: str = Field(..., description="Dynamically identified section title, e.g. 'Laboratory Results'")
    description: str = Field(..., description="Detailed, informative description of all findings under this title")


class TestResultResponse(BaseModel):
    """Top-level response containing array of dynamic test result sections."""
    test_results: List[TestResultSection] = Field(
        ..., description="Array of dynamically identified test result sections"
    )
    input_token: Optional[str] = Field(default=None, description="Number of tokens in the prompt")
    output_token: Optional[str] = Field(default=None, description="Number of tokens in the generated response")
