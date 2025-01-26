from pydantic import BaseModel


class APIResponseBody(BaseModel):
    success: bool
    match_face: str