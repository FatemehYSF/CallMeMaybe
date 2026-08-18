from typing import Literal

from pydantic import BaseModel


class Parameter(BaseModel):
    """Describe one supported function parameter."""

    name: str
    type: Literal["string", "number", "boolean"]


class FunctionDefinition(BaseModel):
    """Describe a function provided by the input definition file."""

    name: str
    description: str
    parameters: list[Parameter]
    return_type: str
