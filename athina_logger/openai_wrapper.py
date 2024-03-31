import importlib
from dataclasses import dataclass
import datetime
import functools
import traceback
import threading
from typing import Any, Callable, Dict, Optional, List
import time
from .athina_meta import AthinaMeta
from .inference_logger import InferenceLogger
from .api_key import AthinaApiKey
import openai
from .util.token_count_helper import get_prompt_tokens_openai_chat_completion, get_completion_tokens_openai_chat_completion

# Check OpenAI version
openai_version = openai.__version__
version_numbers = tuple(map(int, openai_version.split('.')))


def log_to_athina(result: dict, args: dict, athina_meta: AthinaMeta):
    try:
        prompt_slug = "default"
        context = None
        customer_id = None
        customer_user_id = None
        response_time_ms = None
        session_id = None
        user_query = None
        environment = "production"
        external_reference_id = None
        custom_attributes = None
        custom_eval_metrics = None

        if athina_meta:
            prompt_slug = athina_meta.prompt_slug
            context = athina_meta.context
            response_time_ms = athina_meta.response_time
            customer_id = athina_meta.customer_id
            customer_user_id = athina_meta.customer_user_id
            session_id = athina_meta.session_id
            user_query = athina_meta.user_query
            environment = athina_meta.environment or "production"
            external_reference_id = athina_meta.external_reference_id
            custom_attributes = athina_meta.custom_attributes
            custom_eval_metrics = athina_meta.custom_eval_metrics

        InferenceLogger.log_inference(
            prompt_slug=prompt_slug,
            prompt=args["messages"],
            language_model_id=args["model"],
            response=result,
            context=context,
            response_time=response_time_ms,
            customer_id=customer_id,
            customer_user_id=customer_user_id,
            session_id=session_id,
            user_query=user_query,
            environment=environment,
            external_reference_id=external_reference_id,
            custom_attributes=custom_attributes,
            custom_eval_metrics=custom_eval_metrics,
        )
    except Exception as e:
        print("Exception while logging to Athina: ", e)
        traceback.print_exc()


class OpenAiMiddleware:
    _athina_meta: Optional[AthinaMeta]
    _args: Optional[any]
    _kwargs: Optional[dict]
    athina_response = ''

    def __init__(self):
        pass

    def _with_athina_logging(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract args from OpenAI call
            self._args = args
            self._kwargs = kwargs
            self._athina_meta = kwargs.pop("athina_meta", None)

            # Make the OpenAI call and measure response time
            start_time = time.time()
            openai_response = func(*self._args, **self._kwargs)
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)

            # Return if no result was returned from OpenAI
            if openai_response is None:
                print("No result was returned from OpenAI")
                return openai_response

            # Construct the Athina Meta object and log to Athina
            try:
                # Construct the Athina Meta object
                if self._athina_meta is not None:
                    self._athina_meta.response_time = response_time_ms
                else:
                    self._athina_meta = AthinaMeta(
                        prompt_slug="default",
                        response_time=response_time_ms,
                        environment="default",
                    )

                return self._response_interceptor(openai_response, ("stream" in self._kwargs and self._kwargs["stream"]))
            except Exception as e:
                print("Exception in Athina logging: ", e)
                traceback.print_exc()
                return openai_response

        return wrapper

    def _response_interceptor(self, response, is_streaming=False,
                            send_response: Callable[[dict], None] = None):
        def generator_intercept_packets():
            for r in response:
                self.collect_stream_inference_by_chunk(r)
                yield r
            self._log_stream_to_athina()

        if is_streaming:
            return generator_intercept_packets()
        else:
            api_thread = threading.Thread(
                target=log_to_athina,
                kwargs={
                    "result": response if version_numbers < (1, 0, 0) else response.model_dump(),
                    "args": self._kwargs,
                    "athina_meta": self._athina_meta,
                },
            )
            api_thread.start()
            return response

    def _get_text_from_stream_chunk(self, stream_chunk):
        """
        gets the text from the stream chunk
        """
        try:
            text = ''
            choices = stream_chunk.get('choices', []) 
            if choices and len(choices) > 0 and 'delta' in choices[0]:
                delta = choices[0].get('delta', {})
                if 'content' in delta and delta['content'] is not None:
                    text = delta.get('content', '')

            return text
        except Exception as e:
            raise e

    def collect_stream_inference_by_chunk(self, stream_chunk):
        """
        collects the inference from the log stream of openai chat completion chunk by chunk
        """
        try:
            if isinstance(stream_chunk, dict):
                self.athina_response += self._get_text_from_stream_chunk(
                    stream_chunk)
            else:
                self.athina_response += self._get_text_from_stream_chunk(
                    stream_chunk.model_dump())
        except Exception as e:
            raise e

    def _log_stream_to_athina(self):
        """
        logs the stream response to the athina
        """
        try:
            prompt_tokens = self._get_prompt_tokens(
                prompt=self._kwargs["messages"], language_model_id=self._kwargs["model"])
            completion_tokens = self._get_completion_tokens(
                response=self.athina_response, language_model_id=self._kwargs["model"])
            if prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            else:
                total_tokens = None 
            payload = {
                'prompt_slug': self._athina_meta.prompt_slug,
                'prompt': self._kwargs["messages"],
                'language_model_id': self._kwargs["model"],
                'response': self.athina_response,
                'response_time': self._athina_meta.response_time,
                'context': self._athina_meta.context,
                'environment': self._athina_meta.environment,
                'customer_id': str(self._athina_meta.customer_id) if self._athina_meta.customer_id is not None else None,
                'customer_user_id': str(self._athina_meta.customer_user_id) if self._athina_meta.customer_user_id is not None else None,
                'session_id': str(self._athina_meta.session_id) if self._athina_meta.session_id is not None else None,
                'user_query': str(self._athina_meta.user_query) if self._athina_meta.user_query is not None else None,
                'external_reference_id': str(self._athina_meta.external_reference_id) if self._athina_meta.external_reference_id is not None else None,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'custom_attributes': self._athina_meta.custom_attributes,
                'custom_eval_metrics': self._athina_meta.custom_eval_metrics,
            }
            # Remove None fields from the payload
            payload = {k: v for k, v in payload.items() if v is not None}
            InferenceLogger.log_inference(**payload)
        except Exception as e:
            raise e

    def _get_prompt_tokens(self, prompt: List[Dict[str, Any]], language_model_id: str):
        """
        gets the prompt tokens given the prompt for the openai chat model completion
        """
        try:
            tokens = get_prompt_tokens_openai_chat_completion(
                prompt=prompt, language_model_id=language_model_id)
            return tokens
        except Exception as e:
            return None

    def _get_completion_tokens(self, response: str, language_model_id: str):
        """
        gets the completion tokens given the prompt response from the openai chat model completion
        """
        try:
            tokens = get_completion_tokens_openai_chat_completion(
                response=response, language_model_id=language_model_id)
            return tokens
        except Exception as e:
            return None

    # Apply the Athina logging wrapper to OpenAI methods
    def apply_athina(self, openai_instance=None):
        openai_version = openai.__version__
        version_numbers = tuple(map(int, openai_version.split('.')))

        if version_numbers < (1, 0, 0):
            ChatCompletion = importlib.import_module(
                "openai.api_resources").ChatCompletion
            openai_method_name = "create"
            openai_method = getattr(ChatCompletion, openai_method_name)

            # Override the create method with the Athina logging wrapper
            athina_method = self._with_athina_logging(openai_method)
            setattr(ChatCompletion, openai_method_name, athina_method)
        else:
            openai_method_name = "create"
            if openai_instance is not None:
                openai_method = getattr(
                    openai_instance.chat.completions, openai_method_name)

                # Override the chat.completions.create method with the Athina logging wrapper
                athina_method = self._with_athina_logging(openai_method)
                setattr(openai_instance.chat.completions,
                        openai_method_name, athina_method)


middleware = OpenAiMiddleware()
middleware.apply_athina()


if version_numbers > (1, 0, 0):
    # Monkey-patch the constructor of openai.OpenAI
    original_openai_constructor = importlib.import_module(
        "openai").OpenAI.__init__

    def new_openai_constructor(self, *args, **kwargs):
        original_openai_constructor(self, *args, **kwargs)
        middleware.apply_athina(self)

    importlib.import_module(
        "openai").OpenAI.__init__ = new_openai_constructor
