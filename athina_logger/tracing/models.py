
import datetime
from typing import Optional
from pydantic import BaseModel, Field

class TraceModel(BaseModel):
    name: str
    start_time: str = Field(default_factory=datetime.datetime.utcnow)
    end_time: Optional[str] = None
    duration: Optional[int] = None
    status: Optional[str] = None
    attributes: Optional[dict] = None
    version: Optional[str] = None

class SpanModel(BaseModel):
    name: str
    start_time: str = Field(default_factory=datetime.datetime.utcnow)
    span_type: str = "span"
    end_time: Optional[str] = None
    duration: Optional[int] = None
    status: Optional[str] = None
    attributes: Optional[dict] = None
    input: Optional[dict] = None
    output: Optional[dict] = None
    version: Optional[str] = None
