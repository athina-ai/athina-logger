import importlib
from dataclasses import dataclass
import datetime
import functools
import traceback
import threading
from typing import Callable, Optional, List
import time
from .athina_meta import AthinaMeta
from .inference_logger import InferenceLogger
from .api_key import AthinaApiKey
import openai


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
        )
    except Exception as e:
        print("Exception while logging to Athina: ", e)
        traceback.print_exc()


class OpenAiMiddleware:
    _athina_meta: Optional[AthinaMeta]
    _args: Optional[any]
    _kwargs: Optional[dict]

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

                # Log to Athina if it is not a streamed response
                if "stream" not in self._kwargs or not self._kwargs["stream"]:
                    api_thread = threading.Thread(
                        target=log_to_athina,
                        kwargs={
                            "result": openai_response if version_numbers < (1, 0, 0) else openai_response.model_dump(),
                            "args": self._kwargs,
                            "athina_meta": self._athina_meta,
                        },
                    )
                    api_thread.start()

                return openai_response

            except Exception as e:
                print("Exception in Athina logging: ", e)
                traceback.print_exc()
                return openai_response

        return wrapper

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
