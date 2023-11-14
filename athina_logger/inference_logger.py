import requests
from typing import List, Optional, Dict, Union, Any
from .constants import API_BASE_URL
from .api_key import AthinaApiKey
from .request_helper import RequestHelper


class InferenceLogger(AthinaApiKey):
    @staticmethod
    def log_open_ai_chat_response(
        prompt_slug: str,
        messages: List[Dict[str, Any]],
        model: str,
        completion: Optional[Any] = None,
        prompt_response: Optional[str] = None,
        response_time: Optional[int] = None,
        context: Optional[Dict] = None,
        environment: Optional[str] = 'production',
        customer_id: Optional[str] = None,
        customer_user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_query: Optional[str] = None,
        external_reference_id: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        """
        track the request and response for OpenAI chat completion.
        """
        try:
            if completion is None and prompt_response is None:
                raise ValueError(
                    'completion or prompt_response must be provided')

            payload = {
                'prompt_slug': prompt_slug,
                'prompt_messages': messages,
                'language_model_id': model,
                'completion': completion,
                'prompt_response': prompt_response,
                'response_time': response_time,
                'context': context,
                'environment': environment,
                'customer_id': str(customer_id) if customer_id is not None else None,
                'customer_user_id': str(customer_user_id) if customer_user_id is not None else None,
                'session_id': str(session_id) if session_id is not None else None,
                'user_query': str(user_query) if user_query is not None else None,
                'external_reference_id': str(external_reference_id) if external_reference_id is not None else None,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            RequestHelper.make_post_request(endpoint=f'{API_BASE_URL}/api/v1/log/prompt/openai-chat', payload=payload, headers={
                'athina-api-key': InferenceLogger.get_api_key(),
            })
        except Exception as e:
            raise e

    @staticmethod
    def log_open_ai_completion_response(
        prompt_slug: str,
        prompt: str,
        model: str,
        completion: Optional[Any] = None,
        prompt_response: Optional[str] = None,
        response_time: Optional[int] = None,
        context: Optional[Dict] = None,
        environment: Optional[str] = 'production',
        customer_id: Optional[str] = None,
        customer_user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_query: Optional[str] = None,
        external_reference_id: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        """
        track the request and response for OpenAI completion endpoint.
        """
        try:
            if completion is None and prompt_response is None:
                raise ValueError(
                    'completion or prompt_response must be provided')

            payload = {
                'prompt_slug': prompt_slug,
                'prompt_text': prompt,
                'language_model_id': model,
                'completion': completion,
                'prompt_response': prompt_response,
                'response_time': response_time,
                'context': context,
                'environment': environment,
                'customer_id': str(customer_id) if customer_id is not None else None,
                'customer_user_id': str(customer_user_id) if customer_user_id is not None else None,
                'session_id': str(session_id) if session_id is not None else None,
                'user_query': str(user_query) if user_query is not None else None,
                'external_reference_id': str(external_reference_id) if external_reference_id is not None else None,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,

            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            RequestHelper.make_post_request(endpoint=f'{API_BASE_URL}/api/v1/log/prompt/openai-completion', payload=payload, headers={
                'athina-api-key': InferenceLogger.get_api_key(),
            })
        except Exception as e:
            raise e

    @staticmethod
    def log_generic_response(
        prompt_slug: str,
        prompt: str,
        model: str,
        prompt_response: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        response_time: Optional[int] = None,
        context: Optional[Dict] = None,
        environment: Optional[str] = 'production',
        customer_id: Optional[str] = None,
        customer_user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_query: Optional[str] = None,
        external_reference_id: Optional[str] = None,
    ) -> None:
        """
        track a generic language model response
        """
        try:
            payload = {
                'prompt_slug': prompt_slug,
                'prompt_text': prompt,
                'language_model_id': model,
                'prompt_response': prompt_response,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'response_time': response_time,
                'context': context,
                'environment': environment,
                'customer_id': str(customer_id) if customer_id is not None else None,
                'customer_user_id': str(customer_user_id) if customer_user_id is not None else None,
                'session_id': str(session_id) if session_id is not None else None,
                'user_query': str(user_query) if user_query is not None else None,
                'external_reference_id': str(external_reference_id) if external_reference_id is not None else None,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            RequestHelper.make_post_request(endpoint=f'{API_BASE_URL}/api/v1/log/prompt/generic', payload=payload, headers={
                'athina-api-key': InferenceLogger.get_api_key(),
            })
        except Exception as e:
            raise e

    @staticmethod
    def log_langchain_llm_response(
        prompt_slug: str,
        prompt_sent: str,
        prompt_response: str,
        model: str,
        response_time: int,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        context: Optional[Dict] = None,
        environment: Optional[str] = None,
        customer_id: Optional[Union[int, str]] = None,
        customer_user_id: Optional[Union[int, str]] = None,
        session_id: Optional[Union[int, str]] = None,
        user_query: Optional[str] = None,
        external_reference_id: Optional[str] = None,
    ) -> None:
        """
        track the request and response for a language model made by langchain.
        """
        try:
            payload = {
                'prompt_slug': prompt_slug,
                'prompt_sent': prompt_sent,
                'language_model_id': model,
                'prompt_response': prompt_response,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'response_time': response_time,
                'context': context,
                'environment': environment,
                'customer_id': str(customer_id) if customer_id is not None else None,
                'customer_user_id': str(customer_user_id) if customer_user_id is not None else None,
                'session_id': str(session_id) if session_id is not None else None,
                'user_query': str(user_query) if user_query is not None else None,
                'external_reference_id': str(external_reference_id) if external_reference_id is not None else None,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            RequestHelper.make_post_request(endpoint=f'{API_BASE_URL}/api/v1/log/prompt/langchain', payload=payload, headers={
                'athina-api-key': InferenceLogger.get_api_key(),
            })
        except Exception as e:
            raise e
