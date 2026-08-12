from pydantic import BaseModel


class Parameter(BaseModel):
    name: str
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: list[Parameter]
    return_type: str
