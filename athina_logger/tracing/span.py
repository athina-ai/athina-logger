import datetime
from typing import Any, Dict, List, Optional, Union
from .models import SpanModel
from .util import get_utc_time, remove_none_values
from langchain.schema.document import Document

class Span:

    def __init__(
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
    ):
        if start_time is None:
            start_time = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        else:
            start_time = start_time
        self._span = SpanModel(
            name=name,
            span_type=span_type,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat() if end_time else None,
            status=status,
            attributes=attributes or {},
            input=input or {},
            output=output or {},
            duration=duration,
            version=version,
        )
        self._children = []

    def __repr__(self):
        return f"Span(name={self._span.name}, dict={remove_none_values(self.to_dict())}, children={self._children})"

    def to_dict(self):
        span_dict = self._span.model_dump()
        if "input_documents" in self._span.input:
            self._span.input['input_documents'] = [doc.page_content if isinstance(doc, Document) else doc for doc in self._span.input["input_documents"]]
        span_dict["children"] = [child.to_dict() for child in self._children]
        return span_dict

    def add_span(self, span: "Span"):
        self._children.append(span)

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
    ):
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
        self._children.append(span)
        return span

    def add_generation(self, generation: "Generation"):
        self._children.append(generation)

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
    ):
        span = Generation(
            name=name,
            start_time=start_time,
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
        self._children.append(span)
        return span
    
    def update(
        self,
        end_time: Optional[datetime.datetime] = None,
        status: Optional[str] = None,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        duration: Optional[int] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        if end_time:
            self._span.end_time = end_time.replace(tzinfo=datetime.timezone.utc)
        if status:
            self._span.status = status
        if input:
            self._span.input = input
        if output:
            self._span.output = output
        if duration:
            self._span.duration = duration
        if attributes:
            self._span.attributes.update(attributes)

    def end(self, end_time: Optional[datetime.datetime] = None):
        try:
            end_time = get_utc_time(end_time)
            if self._span.end_time is None:
                self._span.end_time = end_time.replace(tzinfo=datetime.timezone.utc).isoformat()
            if self._span.duration is None:
                delta = (end_time - get_utc_time(datetime.datetime.fromisoformat(self._span.start_time)))
                self._span.duration = int((delta.seconds * 1000) + (delta.microseconds // 1000))
            for child in self._children:
                child.end(end_time)
        except Exception as e:
            print(f"Error ending span: {e}")

class Generation(Span):
    def __init__(
        self,
        name: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        span_type: str = "generation",
        status: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
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
    ):
        if attributes is None:
            attributes = {}
        additional_attributes = {
            "prompt": prompt,
            "response": response,
            "prompt_slug": prompt_slug,
            "language_model_id": language_model_id,
            "environment": environment,
            "functions": functions,
            "function_call_response": function_call_response,
            "tools": tools,
            "tool_calls": tool_calls,
            "external_reference_id": external_reference_id,
            "customer_id": customer_id,
            "customer_user_id": customer_user_id,
            "session_id": session_id,
            "user_query": user_query,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "response_time": response_time,
            "context": context,
            "expected_response": expected_response,
            "custom_attributes": custom_attributes,
            "cost": cost,
            "custom_eval_metrics": custom_eval_metrics,
        }
        # Update 'attributes' only with non-None values from 'additional_attributes'
        for key, value in additional_attributes.items():
            if value is not None:
                attributes[key] = value
        super().__init__(
            name=name,
            start_time=start_time,
            end_time=end_time,
            span_type=span_type,
            status=status,
            attributes=attributes,
            input=input,
            output=output,
            duration=duration,
            version=version,
        )

    def update(
        self,
        end_time: Optional[datetime.datetime] = None,
        status: Optional[str] = None,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        duration: Optional[int] = None,
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
    ):
        if self._span.attributes is None:
            self._span.attributes = {}
        attributes = {
            "prompt": prompt if prompt is not None else self._span.attributes.get("prompt"),
            "response": response if response is not None else self._span.attributes.get("response"),
            "prompt_slug": prompt_slug if prompt_slug is not None else self._span.attributes.get("prompt_slug"),
            "language_model_id": language_model_id if language_model_id is not None else self._span.attributes.get("language_model_id"),
            "environment": environment if environment is not None else self._span.attributes.get("environment"),
            "functions": functions if functions is not None else self._span.attributes.get("functions"),
            "function_call_response": function_call_response if function_call_response is not None else self._span.attributes.get("function_call_response"),
            "tools": tools if tools is not None else self._span.attributes.get("tools"),
            "tool_calls": tool_calls if tool_calls is not None else self._span.attributes.get("tool_calls"),
            "external_reference_id": external_reference_id if external_reference_id is not None else self._span.attributes.get("external_reference_id"),
            "customer_id": customer_id if customer_id is not None else self._span.attributes.get("customer_id"),
            "customer_user_id": customer_user_id if customer_user_id is not None else self._span.attributes.get("customer_user_id"),
            "session_id": session_id if session_id is not None else self._span.attributes.get("session_id"),
            "user_query": user_query if user_query is not None else self._span.attributes.get("user_query"),
            "prompt_tokens": prompt_tokens if prompt_tokens is not None else self._span.attributes.get("prompt_tokens"),
            "completion_tokens": completion_tokens if completion_tokens is not None else self._span.attributes.get("completion_tokens"),
            "total_tokens": total_tokens if total_tokens is not None else self._span.attributes.get("total_tokens"),
            "response_time": response_time if response_time is not None else self._span.attributes.get("response_time"),
            "context": context if context is not None else self._span.attributes.get("context"),
            "expected_response": expected_response if expected_response is not None else self._span.attributes.get("expected_response"),
            "custom_attributes": custom_attributes if custom_attributes is not None else self._span.attributes.get("custom_attributes"),
            "cost": cost if cost is not None else self._span.attributes.get("cost"),
            "custom_eval_metrics": custom_eval_metrics if custom_eval_metrics is not None else self._span.attributes.get("custom_eval_metrics"),
        }
        super().update(
            end_time=end_time,
            status=status,
            input=input,
            output=output,
            duration=duration,
            attributes=attributes,
        )
