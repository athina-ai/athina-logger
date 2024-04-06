from abc import ABC, abstractmethod
from typing import Optional, Dict


class LogStreamInference(ABC):
    """
    abstract class for log stream inference.
    """

    def __init__(self,
                 prompt_slug: str,
                 response_time: Optional[int] = None,
                 context: Optional[Dict] = None,
                 environment: Optional[str] = 'production',
                 customer_id: Optional[str] = None,
                 customer_user_id: Optional[str] = None,
                 session_id: Optional[str] = None,
                 user_query: Optional[str] = None,
                 external_reference_id: Optional[str] = None,
                 custom_attributes: Optional[Dict] = None,
                 custom_eval_metrics: Optional[Dict] = None
                ):
        """
        constructor for log stream inference
        :param prompt_slug: str - The slug of the prompt used for the inference.
        :param response_time: Optional[int] - The response time in milliseconds. Defaults to None.
        :param context: Optional[Dict] - A dictionary containing additional context information. Defaults to None.
        :param environment: Optional[str] - The environment in which the inference occurred. Defaults to production.
        :param customer_id: Optional[str] - The customer's unique identifier. Defaults to None.
        :param customer_user_id: Optional[str] - The customer's user identifier. Defaults to None.
        :param session_id: Optional[str] - The session identifier. Defaults to None.
        :param user_query: Optional[str] - The user's query or input. Defaults to None.
        :param external_reference_id: Optional[str] - The external reference id. Defaults to None.
        :param custom_attributes: Optional[Dict] - A dictionary containing custom attributes. Defaults to None.
        :param custom_eval_metrics: Optional[Dict] - A dictionary containing custom evaluation metrics. Defaults to None.
        """
        self._prompt_slug = prompt_slug
        self._response_time = response_time
        self._context = context
        self._environment = environment
        self._customer_id = customer_id
        self._customer_user_id = customer_user_id
        self._session_id = session_id
        self._user_query = user_query
        self._external_reference_id = external_reference_id
        self._custom_attributes = custom_attributes
        self._custom_eval_metrics = custom_eval_metrics

    @property
    def prompt_slug(self):
        return self._prompt_slug

    @prompt_slug.setter
    def prompt_slug(self, value):
        self._prompt_slug = value

    @property
    def response_time(self):
        return self._response_time

    @response_time.setter
    def response_time(self, value):
        self._response_time = value

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        self._context = value

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, value):
        self._environment = value

    @property
    def customer_id(self):
        return self._customer_id

    @customer_id.setter
    def customer_id(self, value):
        self._customer_id = value

    @property
    def customer_user_id(self):
        return self._customer_user_id

    @customer_user_id.setter
    def customer_user_id(self, value):
        self._customer_user_id = value

    @property
    def session_id(self):
        return self._session_id

    @session_id.setter
    def session_id(self, value):
        self._session_id = value

    @property
    def user_query(self):
        return self._user_query

    @user_query.setter
    def user_query(self, value):
        self._user_query = value

    @property
    def external_reference_id(self):
        return self._external_reference_id

    @external_reference_id.setter
    def external_reference_id(self, value):
        self._external_reference_id = value

    @property
    def custom_attributes(self):
        return self._custom_attributes

    @custom_attributes.setter
    def custom_attributes(self, value):
        self._custom_attributes = value
    
    @property
    def custom_eval_metrics(self):
        return self._custom_eval_metrics
    
    @custom_eval_metrics.setter
    def custom_eval_metrics(self, value):
        self._custom_eval_metrics = value

    @abstractmethod
    def collect_stream_inference(self, response):
        """
        collects the inference from the log stream
        """
        pass

    @abstractmethod
    def collect_stream_inference_by_chunk(self, stream_chunk):
        """
        collects the inference from the log stream by chunk
        """
        pass

    @abstractmethod
    def log_stream_inference(self):
        """
        logs the stream inference to the athina api server
        """
        pass
