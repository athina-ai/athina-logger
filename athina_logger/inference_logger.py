import asyncio
import threading
from typing import List, Optional, Dict, Union, Any

from .api_key import AthinaApiKey
from .constants import API_BASE_URL
from .request_helper import RequestHelper


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
            model_options: Optional[dict] = None,
    ) -> None:
        """
            logs prompt run data to athina.

            asynchronously logs structured inference data, safe for performance-critical code.
            errors are suppressed and printed.

            Parameters:
            - prompt (Union[List[Dict[str, Any]], Dict[str, Any], str], optional): The main prompt or message data, supporting string or structured list of messages.
            - response (Any, optional): The response to the prompt, can be of any type.
            - prompt_slug (str, optional): Unique slug identifying the prompt type.
            - language_model_id (str, optional): Identifier for the language model used.
            - environment (str, optional): Operational environment, e.g., 'production' or 'development'.
            - functions (List[Dict[str, Any]], optional): Functions metadata used in processing, as an array of function schemas.
            - function_call_response (Any, optional): Response object or data resulting from function calls.
            - tools (Any, optional): Metadata for tools used in inference, if any.
            - tool_calls (Any, optional): Metadata for tool call results.
            - external_reference_id (str, optional): External reference for tracking, nullable.
            - customer_id (str, optional): Identifier for the customer using the service, nullable.
            - customer_user_id (str, optional): Unique user identifier for the customer’s user, nullable.
            - session_id (str, optional): ID for the current session, nullable.
            - user_query (str, optional): Original query from the user, nullable.
            - prompt_tokens (int, optional): Number of tokens in the prompt input.
            - completion_tokens (int, optional): Number of tokens in the model’s output completion.
            - total_tokens (int, optional): Sum of tokens used (prompt + completion).
            - response_time (int, optional): Time taken to generate a response, in milliseconds.
            - context (Union[Dict[str, Any], str], optional): Additional context data as a dictionary or string, nullable.
            - expected_response (str, optional): The expected response for comparison or validation, nullable.
            - custom_attributes (Dict[str, Any], optional): Additional attributes for custom data, structured as key-value pairs.
            - custom_eval_metrics (Dict[str, Any], optional): Custom evaluation metrics for assessing response quality.
            - cost (float, optional): Inference cost, if calculated.
            - model_options (Dict[str, Any], optional): Configuration options for the model run, including:
              - `temperature` (float, optional): Sampling temperature.
              - `max_completion_tokens` (int, optional): Maximum tokens allowed in response.
              - `stop` (Union[str, List[str]], optional): Stop sequence(s) to halt generation.
              - `top_p` (float, optional): Top-p sampling parameter.
              - `extra_options` (Dict[str, Any], optional): Any additional options for model customization.

            Returns:
            - None: The method does not return any value.

            Raises:
            - None: errors are suppressed and printed.
            """
        try:
            args = (
                prompt, response, prompt_slug, language_model_id, environment, functions, function_call_response, tools,
                tool_calls, external_reference_id, customer_id, customer_user_id, session_id, user_query, prompt_tokens,
                completion_tokens, total_tokens, response_time, context, expected_response, custom_attributes, cost,
                custom_eval_metrics, model_options)
            threading.Thread(target=lambda: asyncio.run(InferenceLogger._log_inference_asynchronously(*args))).start()
        except Exception as e:
            print("Error in logging inference to Athina: ", str(e))

    @staticmethod
    async def _log_inference_asynchronously(
            prompt, response, prompt_slug, language_model_id, environment, functions, function_call_response, tools,
            tool_calls, external_reference_id, customer_id, customer_user_id, session_id, user_query, prompt_tokens,
            completion_tokens, total_tokens, response_time, context, expected_response, custom_attributes, cost,
            custom_eval_metrics, model_options
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
                'model_options': model_options,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            RequestHelper.make_post_request(endpoint=f'{API_BASE_URL}/api/v1/log/inference', payload=payload, headers={
                'athina-api-key': InferenceLogger.get_api_key(),
            })
        except Exception as e:
            print("Error in logging inference to Athina: ", str(e))
