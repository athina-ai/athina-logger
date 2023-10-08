import requests
from typing import Any, List, Optional, TypedDict
from retrying import retry


class RequestHelper:
    """
    class to make requests to the athina api
    """
    @staticmethod
    @retry(wait_fixed=100, stop_max_attempt_number=2)
    def make_post_request(endpoint: str, payload: dict, headers: dict):
        try:
            requests.post(
                endpoint,
                json=payload,
                headers=headers,
            )
        except Exception as e:
            raise e
