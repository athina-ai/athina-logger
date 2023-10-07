from typing import List, Optional, Dict, Any

import requests
from .constants import LOG_OPENAI_COMPLETION_URL
from .log_stream_inference import LogStreamInference
from .api_key import ApiKey


class LogOpenAiCompletionStreamInference(LogStreamInference, ApiKey):
    def __init__(self,
                 prompt_slug: str,
                 prompt: str,
                 model: str,
                 response_time: Optional[int] = None,
                 context: Optional[Dict] = None,
                 environment: Optional[str] = 'production',
                 customer_id: Optional[str] = None,
                 customer_user_id: Optional[str] = None,
                 session_id: Optional[str] = None,
                 user_query: Optional[str] = None,):
        """
        constructor for log stream inference
        :param prompt_slug: str - The slug of the prompt used for the inference.
        :param prompt: str - The prompt used for the inference.
        :param model: str - The model used for the inference.
        :param response_time: Optional[int] - The response time in milliseconds. Defaults to None.
        :param context: Optional[Dict] - A dictionary containing additional context information. Defaults to None.
        :param environment: Optional[str] - The environment in which the inference occurred. Defaults to production.
        :param customer_id: Optional[str] - The customer's unique identifier. Defaults to None.
        :param customer_user_id: Optional[str] - The customer's user identifier. Defaults to None.
        :param session_id: Optional[str] - The session identifier. Defaults to None.
        :param user_query: Optional[str] - The user's query or input. Defaults to None.
        """
        super().__init__(
            prompt_slug=prompt_slug,
            response_time=response_time,
            context=context,
            environment=environment,
            customer_id=customer_id,
            customer_user_id=customer_user_id,
            session_id=session_id,
            user_query=user_query)
        self.prompt = prompt
        self.model = model
        self.prompt_response = ''
        self.log_endpoint = LOG_OPENAI_COMPLETION_URL

    def _get_text_from_stream_chunk(self, stream_chunk):
        """
        gets the text from the stream chunk
        """
        try:
            text = ''
            choices = stream_chunk.get("choices", [])
            if choices and len(choices) > 0 and "text" in choices[0]:
                text = choices[0].get("text", {})

            return text
        except Exception as e:
            raise e

    def collect_stream_inference(self, response):
        """
        collects the inference from the log stream
        """
        try:
            for stream_chunk in response:
                self.prompt_response += self._get_text_from_stream_chunk(
                    stream_chunk)
        except Exception as e:
            raise e

    def collect_stream_inference_by_chunk(self, stream_chunk):
        """
        collects the inference from the log stream of openai chat completion chunk by chunk
        """
        try:
            self.prompt_response += self._get_text_from_stream_chunk(
                stream_chunk)
        except Exception as e:
            raise e

    def log_stream_inference(self):
        """
        logs the stream inference to the athina api server
        """
        try:
            payload = {
                "prompt_slug": self.prompt_slug,
                "prompt_text": self.prompt,
                "language_model_id": self.model,
                "prompt_response": self.prompt_response,
                "response_time": self.response_time,
                "context": self.context,
                "environment": self.environment,
                "customer_id": str(self.customer_id) if self.customer_id is not None else None,
                "customer_user_id": str(self.customer_user_id) if self.customer_user_id is not None else None,
                "session_id": str(self.session_id) if self.session_id is not None else None,
                "user_query": self.user_query,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            requests.post(
                self.log_endpoint,
                json=payload,
                headers={
                    "athina-api-key": self.get_api_key(),
                },
            )
        except Exception as e:
            raise e
