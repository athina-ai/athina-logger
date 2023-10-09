from typing import List, Optional, Dict, Any
from ..constants import LOG_OPENAI_COMPLETION_URL
from .log_stream_inference import LogStreamInference
from ..api_key import AthinaApiKey
from ..request_helper import RequestHelper
from ..util.token_count_helper import get_token_usage_openai_completion


class LogOpenAiCompletionStreamInference(LogStreamInference, AthinaApiKey):
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
                 user_query: Optional[str] = None,
                 external_reference_id: Optional[str] = None,):
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
        :param external_reference_id: Optional[str] - The external reference id. Defaults to None.
        """
        super().__init__(
            prompt_slug=prompt_slug,
            response_time=response_time,
            context=context,
            environment=environment,
            customer_id=customer_id,
            customer_user_id=customer_user_id,
            session_id=session_id,
            user_query=user_query,
            external_reference_id=external_reference_id,)
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
            choices = stream_chunk.get('choices', [])
            if choices and len(choices) > 0 and 'text' in choices[0]:
                text = choices[0].get('text', {})

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
            prompt_tokens = self._get_prompt_tokens(
                prompt=self.prompt, model=self.model)
            completion_tokens = self._get_completion_tokens(
                response=self.prompt_response, model=self.model)
            if prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            else:
                total_tokens = None
            payload = {
                'prompt_slug': self.prompt_slug,
                'prompt_text': self.prompt,
                'language_model_id': self.model,
                'prompt_response': self.prompt_response,
                'response_time': self.response_time,
                'context': self.context,
                'environment': self.environment,
                'customer_id': str(self.customer_id) if self.customer_id is not None else None,
                'customer_user_id': str(self.customer_user_id) if self.customer_user_id is not None else None,
                'session_id': str(self.session_id) if self.session_id is not None else None,
                'user_query': str(self.user_query) if self.user_query is not None else None,
                'external_reference_id': str(self.external_reference_id) if self.external_reference_id is not None else None,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            RequestHelper.make_post_request(endpoint=self.log_endpoint, payload=payload, headers={
                'athina-api-key': LogOpenAiCompletionStreamInference.get_api_key(),
            })
        except Exception as e:
            raise e

    def _get_prompt_tokens(self, prompt: str, model: str):
        """
        gets the prompt tokens given the prompt for the openai chat model completion
        """
        try:
            tokens = get_token_usage_openai_completion(
                text=prompt, model=model)
            return tokens
        except Exception as e:
            return None

    def _get_completion_tokens(self, response: str, model: str):
        """
        gets the completion tokens given the prompt response from the openai chat model completion
        """
        try:
            tokens = get_token_usage_openai_completion(
                text=response, model=model)
            return tokens
        except Exception as e:
            return None
