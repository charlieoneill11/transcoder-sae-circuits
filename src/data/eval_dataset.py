from typing import TypedDict, Optional


class EvalItem(TypedDict):
    text: str
    correct: Optional[str]
    counter: Optional[str]
