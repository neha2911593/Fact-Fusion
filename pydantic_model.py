from pydantic import BaseModel

class ClaimRequest(BaseModel):
    claim: str | None = None
    data: list[str] | None = None
