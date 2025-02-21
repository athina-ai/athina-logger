import functools
import asyncio
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
                    # Capture input if available
                    if args or kwargs:
                        trace._trace.attributes.update({
                            "function_input": {
                                "args": args,
                                "kwargs": kwargs
                            }
                        })
                    
                    result = await func(*args, **kwargs)
                    
                    # Capture output
                    if result is not None:
                        trace._trace.attributes["function_output"] = result
                    
                    return result
                except Exception as e:
                    trace._trace.status = "error"
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
                    # Capture input if available
                    if args or kwargs:
                        trace._trace.attributes.update({
                            "function_input": {
                                "args": args,
                                "kwargs": kwargs
                            }
                        })
                    
                    result = func(*args, **kwargs)
                    
                    # Capture output
                    if result is not None:
                        trace._trace.attributes["function_output"] = result
                    
                    return result
                except Exception as e:
                    trace._trace.status = "error"
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
                    # Capture input if available
                    if args or kwargs:
                        span._span.input = {
                            "args": args,
                            "kwargs": kwargs
                        }
                    
                    result = await func(*args, **kwargs)
                    
                    # Capture output
                    if result is not None:
                        span._span.output = {"result": result}
                    
                    return result
                except Exception as e:
                    span._span.status = "error"
                    span._span.attributes["error"] = str(e)
                    raise
                finally:
                    span.end()
                    current_span.reset(span_token)
                    if trace_token:
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
                    # Capture input if available
                    if args or kwargs:
                        span._span.input = {
                            "args": args,
                            "kwargs": kwargs
                        }
                    
                    result = func(*args, **kwargs)
                    
                    # Capture output
                    if result is not None:
                        span._span.output = {"result": result}
                    
                    return result
                except Exception as e:
                    span._span.status = "error"
                    span._span.attributes["error"] = str(e)
                    raise
                finally:
                    span.end()
                    current_span.reset(span_token)
                    if trace_token:
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
                    # Capture input if available
                    if args or kwargs:
                        generation._span.input = {
                            "args": args,
                            "kwargs": kwargs
                        }
                    
                    result = await func(*args, **kwargs)
                    
                    # Capture output
                    if result is not None:
                        generation._span.output = {"result": result}
                    
                    return result
                except Exception as e:
                    generation._span.status = "error"
                    generation._span.attributes["error"] = str(e)
                    raise
                finally:
                    generation.end()
                    current_span.reset(gen_token)
                    if trace_token:
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
                    # Capture input if available
                    if args or kwargs:
                        generation._span.input = {
                            "args": args,
                            "kwargs": kwargs
                        }
                    
                    result = func(*args, **kwargs)
                    
                    # Capture output
                    if result is not None:
                        generation._span.output = {"result": result}
                    
                    return result
                except Exception as e:
                    generation._span.status = "error"
                    generation._span.attributes["error"] = str(e)
                    raise
                finally:
                    generation.end()
                    current_span.reset(gen_token)
                    if trace_token:
                        trace.end()
                        current_trace.reset(trace_token)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

# Create singleton instance
observe = ObserveDecorator()