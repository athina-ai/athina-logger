
import datetime
from typing import Optional
from pydantic import BaseModel

class TraceCreateModel(BaseModel):
    name: str
    start_time: str = datetime.datetime.now()
    end_time: Optional[str] = None
    duration: Optional[int] = None
    status: Optional[str] = None
    attributes: Optional[dict] = None
    version: Optional[str] = None

class SpanCreateModel(BaseModel):
    name: str
    start_time: str
    span_type: str = "span"
    end_time: Optional[str] = None
    duration: Optional[int] = None
    status: Optional[str] = None
    attributes: Optional[dict] = None
    input: Optional[dict] = None
    output: Optional[dict] = None
    version: Optional[str] = None

# class Trace(BaseModel):
#     id: str
#     trace_type: str
#     start_time: str # optional , if not passed
#     end_time: str
#     duration: int
#     status: str
#     attributes: dict
#     # Athina Fields
#     org_id: str
#     created_at: str
#     updated_at: str


# class Span(BaseModel):
#     id: str
#     trace_id: str
#     parent_id: str
#     span_type: str
#     start_time: str
#     end_time: str
#     duration: int
#     status: str
#     attributes: dict
#     input: dict
#     output: dict
#     created_at: str
#     updated_at: str