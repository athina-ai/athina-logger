import requests
from typing import Optional, Dict, Union
from .constants import API_BASE_URL
from .api_key import ApiKey


class InferenceLogger(ApiKey):
    @staticmethod
    def log_open_ai_chat_response(
        prompt_slug: str,
        messages,
        model: str,
        completion,
        response_time: Optional[int] = None,
        context: Optional[Dict] = None,
        environment: Optional[str] = None,
        customer_id: Optional[str] = None,
        customer_user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_query: Optional[str] = None,
    ) -> None:
        """
        Track the request and response for OpenAI chat completion.
        """
        try:
            payload = {
                'prompt_slug': prompt_slug,
                'prompt_messages': messages,
                'language_model_id': model,
                'completion': completion,
                'response_time': response_time,
                'context': context,
                'environment': environment,
                'customer_id': str(customer_id) if customer_id is not None else None,
                'customer_user_id': str(customer_user_id) if customer_user_id is not None else None,
                'session_id': str(session_id) if session_id is not None else None,
                'user_query': user_query,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            requests.post(
                f'{API_BASE_URL}/api/v1/log/prompt/openai-chat',
                json=payload,
                headers={
                    'athina-api-key': InferenceLogger.get_api_key(),
                },
            )
        except Exception as e:
            raise e

    @staticmethod
    def log_open_ai_completion_response(
        prompt_slug: str,
        prompt: str,
        model: str,
        completion,
        response_time: Optional[int] = None,
        context: Optional[Dict] = None,
        environment: Optional[str] = None,
        customer_id: Optional[str] = None,
        customer_user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_query: Optional[str] = None,
    ) -> None:
        """
        Track the request and response for OpenAI completion endpoint.
        """
        try:
            payload = {
                'prompt_slug': prompt_slug,
                'prompt_text': prompt,
                'language_model_id': model,
                'completion': completion,
                'response_time': response_time,
                'context': context,
                'environment': environment,
                'customer_id': str(customer_id) if customer_id is not None else None,
                'customer_user_id': str(customer_user_id) if customer_user_id is not None else None,
                'session_id': str(session_id) if session_id is not None else None,
                'user_query': user_query,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            requests.post(
                f'{API_BASE_URL}/api/v1/log/prompt/openai-completion',
                json=payload,
                headers={
                    'athina-api-key': InferenceLogger.get_api_key(),
                },
            )
        except Exception as e:
            raise e

    @staticmethod
    def log_generic_response(
        prompt_slug: str,
        prompt: str,
        llm_response: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        response_time: Optional[int] = None,
        context: Optional[Dict] = None,
        environment: Optional[str] = None,
        customer_id: Optional[str] = None,
        customer_user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_query: Optional[str] = None,
    ) -> None:
        """
        Track a generic language model response (not specific to any provider)
        """
        try:
            payload = {
                'prompt_slug': prompt_slug,
                'prompt_text': prompt,
                'language_model_id': 'generic',
                'prompt_response': llm_response,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'response_time': response_time,
                'context': context,
                'environment': environment,
                'customer_id': str(customer_id) if customer_id is not None else None,
                'customer_user_id': str(customer_user_id) if customer_user_id is not None else None,
                'session_id': str(session_id) if session_id is not None else None,
                'user_query': user_query,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            requests.post(
                f'{API_BASE_URL}/api/v1/log/prompt/generic',
                json=payload,
                headers={
                    'athina-api-key': InferenceLogger.get_api_key(),
                },
            )
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
    ) -> None:
        """
        Track the request and response for a language model (not specific to any provider).
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
                'user_query': user_query,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            requests.post(
                f'{API_BASE_URL}/api/v1/log/prompt/langchain',
                json=payload,
                headers={
                    'athina-api-key': InferenceLogger.get_api_key(),
                },
            )
        except Exception as e:
            raise e
