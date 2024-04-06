import requests
from typing import List, Optional, Dict, Union, Any
from .constants import API_BASE_URL
from .api_key import AthinaApiKey
from .request_helper import RequestHelper
import asyncio
import threading

class InferenceLogger(AthinaApiKey):

    @staticmethod
    def log_inference(
        prompt: Optional[Union[List[Dict[str, Any]], Dict[str, Any], str]] = None,
        response: Optional[Any] = None,
        prompt_slug: Optional[str] = None,
        language_model_id: Optional[str] = None,
        environment: Optional[str] = 'production',
        functions: Optional[List[Dict]] = None,
        function_call_response: Optional[Any] = None,
        tools: Optional[Any] = None,
        tool_calls: Optional[Any] = None,
        external_reference_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        customer_user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_query: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        response_time: Optional[int] = None,
        context: Optional[Dict] = None,
        expected_response: Optional[str] = None,
        custom_attributes: Optional[Dict] = None,
        custom_eval_metrics: Optional[Dict] = None,
        cost: Optional[float] = None,
    ) -> None:
        try:
            args = (prompt, response, prompt_slug, language_model_id, environment, functions, function_call_response, tools, tool_calls, external_reference_id, customer_id, customer_user_id, session_id, user_query, prompt_tokens, completion_tokens, total_tokens, response_time, context, expected_response, custom_attributes, cost, custom_eval_metrics)
            threading.Thread(target=lambda: asyncio.run(InferenceLogger._log_inference_asynchronously(*args))).start()
        except Exception as e:
            print("Error in logging inference to Athina: ", str(e))

    @staticmethod
    async def _log_inference_asynchronously(
        prompt, response, prompt_slug, language_model_id, environment, functions, function_call_response, tools, tool_calls, external_reference_id, customer_id, customer_user_id, session_id, user_query, prompt_tokens, completion_tokens, total_tokens, response_time, context, expected_response, custom_attributes, cost, custom_eval_metrics
    ) -> None:
        """
        logs the llm inference to athina
        """
        try:
            payload = {
                'prompt': prompt,
                'response': response,
                'prompt_slug': prompt_slug,
                'language_model_id': language_model_id,
                'functions': functions,
                'function_call_response': function_call_response,
                'tools': tools,
                'tool_calls': tool_calls,
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
                'expected_response': expected_response,
                'custom_attributes': custom_attributes,
                'custom_eval_metrics': custom_eval_metrics,
                'cost': cost,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            RequestHelper.make_post_request(endpoint=f'{API_BASE_URL}/api/v1/log/inference', payload=payload, headers={
                'athina-api-key': InferenceLogger.get_api_key(),
            })
        except Exception as e:
            print("Error in logging inference to Athina: ", str(e))
