import asyncio
import json
import datetime
import threading
from typing import Any, Dict, List, Optional, Union

from .span import Generation, Span
from .models import TraceModel
from .util import get_utc_time, remove_none_values
from athina_logger.api_key import AthinaApiKey
from athina_logger.constants import API_BASE_URL
from athina_logger.request_helper import RequestHelper
from langchain.schema.document import Document

class Trace(AthinaApiKey):
    _trace: TraceModel
    _spans: list[Span]

    def __init__(
        self,
        name: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        status: Optional[str] = None,
        attributes: Optional[dict] = None,
        duration: Optional[int] = None,
        version: Optional[str] = None,
    ):        
        self._trace = TraceModel(
            name=name,
            start_time=(start_time or datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)).isoformat(),
            end_time=end_time.isoformat() if end_time else None,
            status=status,
            attributes=attributes or {},
            duration=duration,
            version=version,
        )
        self._spans = []

    def __repr__(self):
        return f"Trace(name={self._trace.name}, dict={remove_none_values(self.to_dict())},  spans={self._spans})"

    def add_span(self, span: Span):
        self._spans.append(span)

    def create_span(
        self,
        name: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        span_type: str = "span",
        status: Optional[str] = None,
        attributes: Optional[dict] = None,
        input: Optional[dict] = None,
        output: Optional[dict] = None,
        duration: Optional[int] = None,
        version: Optional[str] = None,
    )-> Span:
        span = Span(
            name=name,
            start_time=(start_time or datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)),
            end_time=end_time,
            span_type=span_type,
            status=status,
            attributes=attributes or {},
            input=input or {},
            output=output or {},
            duration=duration,
            version=version,
        )
        self._spans.append(span)
        return span

    def add_generation(self, generation: Generation):
        self._spans.append(generation)

    def create_generation(
        self,
        name: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        span_type: str = "generation",
        status: Optional[str] = None,
        attributes: Optional[dict] = None,
        input: Optional[dict] = None,
        output: Optional[dict] = None,
        duration: Optional[int] = None,
        version: Optional[str] = None,
        prompt: Optional[Union[List[Dict[str, Any]], Dict[str, Any], str]] = None,
        response: Optional[Any] = None,
        prompt_slug: Optional[str] = None,
        language_model_id: Optional[str] = None,
        environment: Optional[str] = None,
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
        cost: Optional[float] = None,
        custom_eval_metrics: Optional[Dict] = None,
    ) -> Generation:
        span = Generation(
            name=name,
            start_time=(start_time or datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)),
            end_time=end_time,
            span_type=span_type,
            status=status,
            attributes=attributes,
            input=input,
            output=output,
            duration=duration,
            version=version,
            prompt=prompt,
            response=response,
            prompt_slug=prompt_slug,
            language_model_id=language_model_id,
            environment=environment,
            functions=functions,
            function_call_response=function_call_response,
            tools=tools,
            tool_calls=tool_calls,
            external_reference_id=external_reference_id,
            customer_id=customer_id,
            customer_user_id=customer_user_id,
            session_id=session_id,
            user_query=user_query,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_time=response_time,
            context=context,
            expected_response=expected_response,
            custom_attributes=custom_attributes,
            cost=cost,
            custom_eval_metrics=custom_eval_metrics,
        )
        self._spans.append(span)
        return span
    
    def to_dict(self):
        trace_dict = self._trace.model_dump()
        trace_dict["spans"] = [span.to_dict() for span in self._spans]
        return trace_dict

    def update(
        self,
        end_time: Optional[datetime.datetime] = None,
        duration: Optional[int] = None,
        status: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        if end_time:
            self._trace.end_time = end_time.replace(tzinfo=datetime.timezone.utc).isoformat()
        if status:
            self._trace.status = status
        if attributes:
            self._trace.attributes = attributes
        if duration:
            self._trace.duration = duration

    async def _log_trace_async(self, request_dict: Dict[str, Any]):
        RequestHelper.make_post_request(endpoint=f'{API_BASE_URL}/api/v1/trace/sdk', payload=request_dict, headers={
            "athina-api-key": Trace.get_api_key(), "Content-Type": "application/json"
        })

    def end(self, end_time: Optional[datetime.datetime] = None):
        try:      
            end_time = get_utc_time(end_time)
            for span in self._spans:
                span.end(end_time)
            if self._trace.end_time is None:
                self._trace.end_time =  end_time.replace(tzinfo=datetime.timezone.utc).isoformat()
            if self._trace.duration is None:
                delta = (end_time - get_utc_time(datetime.datetime.fromisoformat(self._trace.start_time)))
                self._trace.duration = int((delta.seconds * 1000) + (delta.microseconds // 1000))
            request_dict = remove_none_values(self.to_dict())
            threading.Thread(target=lambda: asyncio.run(self._log_trace_async(request_dict))).start()
        except Exception as e:
            print("Error ending trace: ", e)