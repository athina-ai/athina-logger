from dataclasses import dataclass
from typing import Optional


@dataclass
class AthinaMeta:
    prompt_slug: Optional[str] = None
    context: Optional[dict] = None
    customer_id: Optional[dict] = None
    customer_user_id: Optional[dict] = None
    session_id: Optional[dict] = None
    user_query: Optional[dict] = None
    environment: Optional[dict] = None
    external_reference_id: Optional[dict] = None
    customer_id: Optional[str] = None
    customer_user_id: Optional[str] = None
    response_time: Optional[int] = None
    custom_attributes: Optional[dict] = None
