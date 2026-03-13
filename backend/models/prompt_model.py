from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PromptTemplate:
    name: str
    template: str
    description: str
    input_variables: list[str]
    version: str = "1.0"
    metadata: Optional[Dict[str, Any]] = None

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)
