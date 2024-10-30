from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class AthinaMeta:
    prompt_slug: Optional[str] = None
    context: Optional[Dict] = None
    session_id: Optional[str] = None
    user_query: Optional[str] = None
    environment: Optional[str] = 'production'
    external_reference_id: Optional[str] = None
    customer_id: Optional[str] = None
    customer_user_id: Optional[str] = None
    response_time: Optional[int] = None
    custom_attributes: Optional[Dict] = None
    custom_eval_metrics: Optional[Dict] = None