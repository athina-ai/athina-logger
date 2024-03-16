import json
from time import sleep
from constants import API_BASE_URL
from api_key import AthinaApiKey
from request_helper import RequestHelper
import requests
import datetime
from typing import Optional
from models import SpanCreateModel, TraceCreateModel

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
    ):
        if start_time is None:
            start_time = datetime.datetime.now()
        if attributes is None:
            attributes = {}
        if input is None:
            input = {}
        if output is None:
            output = {}
        
        self._span = SpanCreateModel(
            name=name,
            span_type=span_type,
            start_time=start_time.isoformat(),
            end_time=end_time,
            status=status,
            attributes=attributes,
            input=input,
            output=output,
            duration=duration,
        )
        self._children = []

    def __repr__(self):
        return f"Span(name={self._span.name}, children={self._children})"

    def to_dict(self):
        return {
            "span": self._span.model_dump(),
            "children": [span.to_dict() for span in self._children]
        }
    
    def span(
        self,
        name: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        span_type: str = "span",
        status: Optional[str] = None,
        attributes: Optional[dict] = None,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        duration: Optional[int] = None,
    ):
        span = Span(
            name=name,
            start_time=start_time or datetime.datetime.now(),
            end_time=end_time,
            span_type=span_type,
            status=status,
            attributes=attributes or {},
            input=input_data or {},
            output=output_data or {},
            duration=duration,
        )
        self._children.append(span)
        return span

    def generation(
        self,
        name: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        status: Optional[str] = None,
        attributes: Optional[dict] = None,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        duration: Optional[int] = None,
    ):
        return self.span(
            name=name,
            start_time=start_time,
            end_time=end_time,
            span_type="generation",
            status=status,
            attributes=attributes,
            input_data=input_data,
            output_data=output_data,
            duration=duration,
        )   

    def end(self, end_time: Optional[datetime.datetime] = None):
        if end_time is None:
            end_time = datetime.datetime.now()
        self._span.end_time = end_time.isoformat()
        self._span.duration = int((end_time - datetime.datetime.fromisoformat(self._span.start_time)).total_seconds())
        for child in self._children:
            if child._span.end_time is None:
                child.end(end_time)            

class Trace(AthinaApiKey):
    def __init__(
        self,
        name: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        status: Optional[str] = None,
        attributes: Optional[dict] = None,
        duration: Optional[int] = None,
    ):
        if start_time is None:
            start_time = datetime.datetime.now() 
        
        self._trace = TraceCreateModel(
            name=name,
            start_time=start_time.isoformat(),
            end_time=end_time,
            status=status,
            attributes=attributes,
            duration=duration,
        )
        self._spans = []

    def __repr__(self):
        return f"Trace(name={self._trace.name}, spans={self._spans})"

    def span(
        self,
        name: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        span_type: str = "span",
        status: Optional[str] = None,
        attributes: Optional[dict] = None,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        duration: Optional[int] = None,
    )-> Span:
        if start_time is None:
            start_time = datetime.datetime.now()
        if attributes is None:
            attributes = {}
        if input_data is None:
            input_data = {}
        if output_data is None:
            output_data = {}

        span = Span(
            name=name,
            start_time=start_time,
            end_time=end_time,
            span_type=span_type,
            status=status,
            attributes=attributes,
            input=input_data,
            output=output_data,
            duration=duration,
        )
        self._spans.append(span)
        return span
    
    def to_dict(self):
        return {
            "trace": self._trace.model_dump(),
            "spans": [span.to_dict() for span in self._spans]
        }

    def end(self):
        for span in self._spans:
            if span._span.end_time is None:
                span.end()
        if self._trace.end_time is None:
            end_time = datetime.datetime.now()
            self._trace.end_time = end_time.isoformat()
            self._trace.duration = int((end_time - datetime.datetime.fromisoformat(self._trace.start_time)).total_seconds())
        data = json.dumps(self.to_dict())
        # response = requests.post("https://8e714940905f4022b43267e348b8a713.api.mockbin.io/", headers= {'Content-Type': 'application/json'}, data=data)
        response = RequestHelper.make_post_request(endpoint=f'{API_BASE_URL}/api/v1/log/inference', payload=data, headers={
            'athina-api-key': Trace.get_api_key(),
        })
        if response.status_code == 200:
            print("Trace and spans successfully saved.")
        else:
            print(f"Failed to save trace and spans. Status Code: {response.status_code}")

if __name__ == "__main__":
    # Create a Trace
    print("\nStarting Trace:")
    trace = Trace(name="trace_1")
    print(trace)

    print("\nAssociate Spans with the Trace")
    span1 = trace.span(name="span_1")
    span2 = trace.span(name="span_2")
    print("\nCurrent Trace:")
    print(trace)

    print("\nAdd a child Span to one of the Spans")
    print(span1)
    span1.span(name="span_3")
    print(span1)

    print("\nAdd a child Span(Generation type) to one of the Spans")
    print(span2)
    span2.generation(name="generation_1")
    print(span2)

    print("\nFinal Trace:")
    print(trace)
    sleep(2)
    trace.end()
    print("\nEnd the Trace")