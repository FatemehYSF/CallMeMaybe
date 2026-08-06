from dataclasses import dataclass
from typing import List


@dataclass
class Parameter:
    name: str
    type: str


@dataclass
class FunctionDefinition:
    name: str
    description: str
    parameters: List[Parameter]
    return_type: str
