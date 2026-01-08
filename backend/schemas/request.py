from pydantic import BaseModel

class TextInput(BaseModel):
    text: str
    top_k: int = 3
