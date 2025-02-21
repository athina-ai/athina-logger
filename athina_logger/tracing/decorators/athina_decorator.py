import functools
import asyncio
import inspect
from typing import Optional, Dict, Any, Callable, TypeVar
from contextvars import ContextVar

from ..trace import Trace

# Context variables to maintain state
current_trace = ContextVar('current_trace', default=None)
current_span = ContextVar('current_span', default=None)

F = TypeVar('F', bound=Callable)

class ObserveDecorator:
    
    def trace(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
    ):
        def decorator(func: F) -> F:
            # Get the signature of the function
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                trace_name = name or func.__name__
                trace = Trace(
                    name=trace_name,
                    attributes=attributes,
                    version=version
                )
                
                # Set trace in context
                token = current_trace.set(trace)
                try:
                    # Map positional args to their parameter names
                    args_dict = {}
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            args_dict[param_names[i]] = arg
                    
                    # Combine with keyword args
                    input_dict = {**args_dict, **kwargs}
                    
                    # Capture input if available
                    if input_dict:
                        trace._trace.input = input_dict
                    
                    result = await func(*args, **kwargs)
                    
                    # Capture output
                    if result is not None:
                        trace._trace.output = {"result": result}
                    
                    # Set success status if not already set
                    if trace._trace.status is None:
                        trace._trace.status = "success"
                        
                    return result
                except Exception as e:
                    trace._trace.status = "error"
                    if trace._trace.attributes is None:
                        trace._trace.attributes = {}
                    trace._trace.attributes["error"] = str(e)
                    raise
                finally:
                    trace.end()
                    current_trace.reset(token)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                trace_name = name or func.__name__
                trace = Trace(
                    name=trace_name,
                    attributes=attributes,
                    version=version
                )
                
                # Set trace in context
                token = current_trace.set(trace)
                try:
                    # Map positional args to their parameter names
                    args_dict = {}
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            args_dict[param_names[i]] = arg
                    
                    # Combine with keyword args
                    input_dict = {**args_dict, **kwargs}
                    
                    # Capture input if available
                    if input_dict:
                        trace._trace.input = input_dict
                    
                    result = func(*args, **kwargs)
                    
                    # Capture output
                    if result is not None:
                        trace._trace.output = {"result": result}
                    
                    # Set success status if not already set
                    if trace._trace.status is None:
                        trace._trace.status = "success"
                        
                    return result
                except Exception as e:
                    trace._trace.status = "error"
                    if trace._trace.attributes is None:
                        trace._trace.attributes = {}
                    trace._trace.attributes["error"] = str(e)
                    raise
                finally:
                    trace.end()
                    current_trace.reset(token)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def span(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        span_type: str = "span",
        version: Optional[str] = None,
    ):
        def decorator(func: F) -> F:
            # Get the signature of the function
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                span_name = name or func.__name__
                
                # Get current trace or create new one if none exists
                trace = current_trace.get()
                trace_token = None
                if trace is None:
                    trace = Trace(span_name, attributes=attributes)
                    trace_token = current_trace.set(trace)
                
                # Check if we're inside another span
                parent_span = current_span.get()
                
                # Create and add span
                if parent_span is not None:
                    # If inside another span, create child span
                    span = parent_span.create_span(
                        name=span_name,
                        span_type=span_type,
                        attributes=attributes,
                        version=version
                    )
                else:
                    # Otherwise create a span at the trace level
                    span = trace.create_span(
                        name=span_name,
                        span_type=span_type,
                        attributes=attributes,
                        version=version
                    )
                
                # Set span in context
                span_token = current_span.set(span)
                
                try:
                    # Map positional args to their parameter names
                    args_dict = {}
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            args_dict[param_names[i]] = arg
                    
                    # Combine with keyword args
                    input_dict = {**args_dict, **kwargs}
                    
                    # Capture input if available
                    if input_dict:
                        span._span.input = input_dict
                    
                    result = await func(*args, **kwargs)
                    
                    # Capture output
                    if result is not None:
                        span._span.output = {"result": result}
                    
                    # Set success status if not already set
                    if span._span.status is None:
                        span._span.status = "success"
                        
                    return result
                except Exception as e:
                    span._span.status = "error"
                    if span._span.attributes is None:
                        span._span.attributes = {}
                    span._span.attributes["error"] = str(e)
                    raise
                finally:
                    span.end()
                    current_span.reset(span_token)
                    if trace_token:
                        # Set success status on trace if not already set
                        if trace._trace.status is None:
                            trace._trace.status = "success"
                        trace.end()
                        current_trace.reset(trace_token)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                span_name = name or func.__name__
                
                # Get current trace or create new one if none exists
                trace = current_trace.get()
                trace_token = None
                if trace is None:
                    trace = Trace(span_name, attributes=attributes)
                    trace_token = current_trace.set(trace)
                
                # Check if we're inside another span
                parent_span = current_span.get()
                
                # Create and add span
                if parent_span is not None:
                    # If inside another span, create child span
                    span = parent_span.create_span(
                        name=span_name,
                        span_type=span_type,
                        attributes=attributes,
                        version=version
                    )
                else:
                    # Otherwise create a span at the trace level
                    span = trace.create_span(
                        name=span_name,
                        span_type=span_type,
                        attributes=attributes,
                        version=version
                    )
                
                # Set span in context
                span_token = current_span.set(span)
                
                try:
                    # Map positional args to their parameter names
                    args_dict = {}
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            args_dict[param_names[i]] = arg
                    
                    # Combine with keyword args
                    input_dict = {**args_dict, **kwargs}
                    
                    # Capture input if available
                    if input_dict:
                        span._span.input = input_dict
                    
                    result = func(*args, **kwargs)
                    
                    # Capture output
                    if result is not None:
                        span._span.output = {"result": result}
                    
                    # Set success status if not already set
                    if span._span.status is None:
                        span._span.status = "success"
                        
                    return result
                except Exception as e:
                    span._span.status = "error"
                    if span._span.attributes is None:
                        span._span.attributes = {}
                    span._span.attributes["error"] = str(e)
                    raise
                finally:
                    span.end()
                    current_span.reset(span_token)
                    if trace_token:
                        # Set success status on trace if not already set
                        if trace._trace.status is None:
                            trace._trace.status = "success"
                        trace.end()
                        current_trace.reset(trace_token)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def generation(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None
    ):
        def decorator(func: F) -> F:
            # Get the signature of the function
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                gen_name = name or func.__name__
                
                # Get current trace or create new one if none exists
                trace = current_trace.get()
                trace_token = None
                if trace is None:
                    trace = Trace(gen_name, attributes=attributes)
                    trace_token = current_trace.set(trace)
                
                # Check if we're inside another span
                parent_span = current_span.get()
                
                # Create and add generation
                if parent_span is not None:
                    # If inside another span, create child generation
                    generation = parent_span.create_generation(
                        name=gen_name,
                        attributes=attributes,
                        version=version
                    )
                else:
                    # Otherwise create a generation at the trace level
                    generation = trace.create_generation(
                        name=gen_name,
                        attributes=attributes,
                        version=version
                    )
                
                # Set generation in context
                gen_token = current_span.set(generation)
                
                try:
                    # Map positional args to their parameter names
                    args_dict = {}
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            args_dict[param_names[i]] = arg
                    
                    # Combine with keyword args
                    input_dict = {**args_dict, **kwargs}
                    
                    # Capture input if available
                    if input_dict:
                        generation._span.input = input_dict
                    
                    result = await func(*args, **kwargs)
                    
                    # Capture output
                    if result is not None:
                        generation._span.output = {"result": result}
                    
                    # Set success status if not already set
                    if generation._span.status is None:
                        generation._span.status = "success"
                        
                    return result
                except Exception as e:
                    generation._span.status = "error"
                    if generation._span.attributes is None:
                        generation._span.attributes = {}
                    generation._span.attributes["error"] = str(e)
                    raise
                finally:
                    generation.end()
                    current_span.reset(gen_token)
                    if trace_token:
                        # Set success status on trace if not already set
                        if trace._trace.status is None:
                            trace._trace.status = "success"
                        trace.end()
                        current_trace.reset(trace_token)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                gen_name = name or func.__name__
                
                # Get current trace or create new one if none exists
                trace = current_trace.get()
                trace_token = None
                if trace is None:
                    trace = Trace(gen_name, attributes=attributes)
                    trace_token = current_trace.set(trace)
                
                # Check if we're inside another span
                parent_span = current_span.get()
                
                # Create and add generation
                if parent_span is not None:
                    # If inside another span, create child generation
                    generation = parent_span.create_generation(
                        name=gen_name,
                        attributes=attributes,
                        version=version
                    )
                else:
                    # Otherwise create a generation at the trace level
                    generation = trace.create_generation(
                        name=gen_name,
                        attributes=attributes,
                        version=version
                    )
                
                # Set generation in context
                gen_token = current_span.set(generation)
                
                try:
                    # Map positional args to their parameter names
                    args_dict = {}
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            args_dict[param_names[i]] = arg
                    
                    # Combine with keyword args
                    input_dict = {**args_dict, **kwargs}
                    
                    # Capture input if available
                    if input_dict:
                        generation._span.input = input_dict
                    
                    result = func(*args, **kwargs)
                    
                    # Capture output
                    if result is not None:
                        generation._span.output = {"result": result}
                    
                    # Set success status if not already set
                    if generation._span.status is None:
                        generation._span.status = "success"
                        
                    return result
                except Exception as e:
                    generation._span.status = "error"
                    if generation._span.attributes is None:
                        generation._span.attributes = {}
                    generation._span.attributes["error"] = str(e)
                    raise
                finally:
                    generation.end()
                    current_span.reset(gen_token)
                    if trace_token:
                        # Set success status on trace if not already set
                        if trace._trace.status is None:
                            trace._trace.status = "success"
                        trace.end()
                        current_trace.reset(trace_token)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def update_current_span(
        self, 
        input: Optional[Any] = None, 
        output: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None
    ):
        """Update the current span with new information."""
        span = current_span.get()
        if span and span._span:
            if input is not None:
                span._span.input = input
            
            if output is not None:
                span._span.output = output
            
            if attributes is not None:
                if span._span.attributes is None:
                    span._span.attributes = {}
                span._span.attributes.update(attributes)
            
            if status is not None:
                span._span.status = status
    
    def update_current_trace(
        self, 
        input: Optional[Any] = None, 
        output: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None
    ):
        """Update the current trace with new information."""
        trace = current_trace.get()
        if trace and trace._trace:
            if input is not None:
                trace._trace.input = input
            
            if output is not None:
                trace._trace.output = output
            
            if attributes is not None:
                if trace._trace.attributes is None:
                    trace._trace.attributes = {}
                trace._trace.attributes.update(attributes)
            
            if status is not None:
                trace._trace.status = status

    def get_current_trace(self):
        """Get the current trace object."""
        return current_trace.get()
    
    def get_current_span(self):
        """Get the current span object."""
        return current_span.get()

# Create singleton instance
observe = ObserveDecorator()