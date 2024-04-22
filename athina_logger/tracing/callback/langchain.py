from langchain.globals import get_debug
from athina_logger.api_key import AthinaApiKey
 
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID, uuid4
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.prompt_values import ChatPromptValue
from langchain.schema.agent import AgentAction, AgentActionMessageLog, AgentFinish
from langchain.schema.document import Document
from langchain_core.outputs import (
    ChatGeneration,
    LLMResult,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    FunctionMessage,
)
from athina_logger.tracing.trace import Trace
from athina_logger.util.extract_model import _extract_model_name

class LangchainCallbackHandler(
    BaseCallbackHandler, AthinaApiKey
):
    def __init__(
        self,
        trace_name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None: 
        _debug("LangchainCallbackHandler.__init__")
        self.version = version
        self.trace_name = trace_name
        self.trace_run_id = None
        self.runs = {}
        self.trace: Trace = None 

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        _debug(
            f"on llm new token: run_id: {run_id} parent_run_id: {parent_run_id}"
        )

    def _get_athina_run_name(self, serialized: Dict[str, Any], **kwargs: Any) -> str:
        """
        Retrieves the 'run_name' for an entity prioritizing the 'name' key in 'kwargs' or falling
        back to the 'name' or 'id' in 'serialized'. Defaults to "<unknown>" if none are available.

        Args:
            serialized (Dict[str, Any]): A dictionary containing the entity's serialized data.
            **kwargs (Any): Additional keyword arguments, potentially including the 'name' override.

        Returns:
            str: The determined athina run name for the entity.
        """
        # Check if 'name' is in kwargs and not None, otherwise use default fallback logic
        if "name" in kwargs and kwargs["name"] is not None:
            return str(kwargs["name"])

        # Fallback to serialized 'name', 'id', or "<unknown>"
        return serialized.get("name", serialized.get("id", ["<unknown>"])[-1])

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any) -> Any:
        _debug(
            f"on chain start: run_id: {run_id} parent_run_id: {parent_run_id}, name {serialized.get('name', serialized.get('id', ['<unknown>'])[-1])}"
        )
        self._generate_trace(
            serialized=serialized,
            inputs=inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            version=self.version,
            **kwargs,
        )
        name = self._get_athina_run_name(serialized, **kwargs) 
        if "intermediate_steps" in inputs:
            inputs["intermediate_steps"] = [
                { 
                    "tool_input":agent_action_message.tool_input,
                    "log":agent_action_message.log
                } for agent_action_message in inputs["intermediate_steps"]
            ]
        if parent_run_id is None:
            self.runs[run_id] = self.trace.create_span(name=name, input=inputs, version=self.version)

        if parent_run_id is not None:
            self.runs[run_id] = self.runs[parent_run_id].create_span(name=name, input=inputs, version=self.version) 

    def convert_outputs_to_dict(self, outputs: Dict[str, Any]) -> Dict:
        if "intermediate_steps" in outputs:
            outputs["intermediate_steps"] = [
                { 
                    "tool_input":agent_action_message.tool_input,
                    "log":agent_action_message.log
                } for agent_action_message in outputs["intermediate_steps"]
            ]
        if type(outputs) == AgentActionMessageLog:
            outputs = { 
                "tool_input": outputs.tool_input,
                "log": outputs.log
            }
        if type(outputs) == ChatPromptValue:
            outputs = { 
                "messages": outputs.to_string(),
            }
        if type(outputs) == AgentFinish:
            outputs = { 
                "return_values": outputs.return_values,
            }
        if type(outputs) == FunctionMessage:
            outputs = {
                "content": outputs.content
            }
        if type(outputs) == list and all(isinstance(item, AIMessageChunk) for item in outputs):
            outputs = [
                { 
                    "type":ai_message_chunk.type
                } for ai_message_chunk in outputs
            ]
        if "agent_scratchpad" in outputs:
            outputs["agent_scratchpad"] = [
                { 
                    "type":ai_message_chunk.type
                } for ai_message_chunk in outputs["agent_scratchpad"]
            ]
        if outputs is None or type(outputs) is not dict:
            outputs = {}
        return outputs

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            _debug(
                f"on chain end: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            if run_id not in self.runs:
                raise Exception("run not found")
            outputs = self.convert_outputs_to_dict(outputs)
            self._update_run(run_id, outputs, None)
            self.runs[run_id].end()
            if self.trace_run_id == run_id:
                self.trace.end()
        except Exception as e:
            _debug(e)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            _debug(
                f"on chain error: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            self._update_run(run_id, None, error)
            self.runs[run_id].end()

        except Exception as e:
            _debug(e)

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action."""
        try:
            _debug(
                f"on agent action: run_id: {run_id} parent_run_id: {parent_run_id}"
            )

            if run_id not in self.runs:
                raise Exception("run not found")

            self._update_run(run_id, {"agent_action" : action.tool_input}, None)
            self.runs[run_id].end()

        except Exception as e:
            _debug(e)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            _debug(
                f"on agent finish: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            if run_id not in self.runs:
                raise Exception("run not found")

            self._update_run(run_id, {"agent_finish": finish.return_values}, None)
            self.runs[run_id].end()

        except Exception as e:
            _debug(e)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            _debug(
                f"on chat model start: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            if metadata is None:
                metadata = {}
            metadata['is_chat_model'] = True
            self._on_llm_action(
                serialized,
                run_id,
                _flatten_comprehension(
                    [self._create_message_dicts(m) for m in messages]
                ),
                parent_run_id,
                tags=tags,
                metadata=metadata,
                **kwargs,
            )
        except Exception as e:
            _debug(e)

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            _debug(
                f"on retriever start: run_id: {run_id} parent_run_id: {parent_run_id}"
            )

            if parent_run_id is None or parent_run_id not in self.runs:
                raise Exception("parent run not found")

            self.runs[run_id] = self.runs[parent_run_id].create_span(
                name=self._get_athina_run_name(serialized, **kwargs),
                input={"query":query},
                attributes=self._join_tags_and_metadata(tags, metadata),
                version=self.version,
            )

        except Exception as e:
            _debug(e)

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            _debug(
                f"on retriever end: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")
            documents_json = {"documents" : [doc.page_content for doc in documents]}
            self._update_run(run_id, documents_json, None)
            self.runs[run_id].end()
        except Exception as e:
            _debug(e)

    def on_retriever_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever errors."""
        try:
            _debug(
                f"on retriever error: run_id: {run_id} parent_run_id: {parent_run_id}"
            )

            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")
            self._update_run(run_id, None, error)
            self.runs[run_id].end()

        except Exception as e:
            _debug(e)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            _debug(
                f"on tool start: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            if parent_run_id is None or parent_run_id not in self.runs:
                raise Exception("parent run not found")
            self.runs[run_id] = self.runs[parent_run_id].create_span(
                name=self._get_athina_run_name(serialized, **kwargs),
                input={"input_str": input_str},
                attributes=self._join_tags_and_metadata(tags, metadata),
                version=self.version,
            )
        except Exception as e:
            _debug(e)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            _debug(
                f"on tool end: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")
            self._update_run(run_id, {"tool_output" : output}, None)
            self.runs[run_id].end()

        except Exception as e:
            _debug(e)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            _debug(
                f"on tool error: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")
            self._update_run(run_id, None, error)
            self.runs[run_id].end()

        except Exception as e:
            _debug(e)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            _debug(
                f"on llm start: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            _debug("metadata:")
            _debug(metadata)
            self._on_llm_action(
                serialized,
                run_id,
                prompts[0] if len(prompts) == 1 else prompts,
                parent_run_id,
                tags=tags,
                metadata=metadata,
                **kwargs,
            )
        except Exception as e:
            _debug(e)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            _debug(
                f"on llm end: run_id: {run_id} parent_run_id: {parent_run_id} response: {response} kwargs: {kwargs}"
            )
            if run_id not in self.runs:
                raise Exception("Run not found, something went wrong.")
            else:
                generation = response.generations[-1][-1]
                extracted_response = (
                    self._convert_message_to_dict(generation.message)
                    if isinstance(generation, ChatGeneration)
                    else _extract_raw_response(generation)
                )
                if isinstance(extracted_response, Dict):
                    extracted_response = extracted_response.get('content', None)

                llm_usage = (
                    None
                    if response.llm_output is None
                    or not response.llm_output["token_usage"]
                    else response.llm_output["token_usage"]
                )
                self.runs[run_id].update(
                    prompt_tokens=llm_usage["prompt_tokens"] if llm_usage else None, 
                    completion_tokens=llm_usage["completion_tokens"] if llm_usage else None,
                    total_tokens=llm_usage["total_tokens"] if llm_usage else None, 
                    response=extracted_response
                )
                self.runs[run_id].end()
        except Exception as e:
            import traceback
            traceback.print_exc()
            _debug(e)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            _debug(
                f"on llm error: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            self._update_run(run_id, None, error)
            self.runs[run_id].end()

        except Exception as e:
            _debug(e)

    def _on_llm_action(
        self,
        serialized: Dict[str, Any],
        run_id: UUID,
        prompts: List[Any],
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        try:
            self._generate_trace(
                serialized,
                inputs=prompts[0] if len(prompts) == 1 else prompts,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
                metadata=metadata,
                version=self.version,
                kwargs=kwargs,
            ) 
            name = self._get_athina_run_name(serialized, **kwargs)
            attributes = { 
                'is_chat_model' : metadata.get('is_chat_model', False),
                'prompt_slug': metadata.get('prompt_slug', None),
                'user_query': metadata.get('user_query', None),
                'context': metadata.get('global_context', None),
                'prompt': prompts,
                'session_id': metadata.get('session_id', None),
                'customer_id': metadata.get('customer_id', None),
                'customer_user_id': metadata.get('customer_user_id', None),
                'external_reference_id': metadata.get('external_reference_id', None),
                'custom_attributes': metadata.get('custom_attributes', None),
                'language_model_id': _extract_model_name(serialized, **kwargs)
            }
            prompt_slug = metadata.get('prompt_slug', None)
            _debug(f"Creating generation with name: {name} and attributes: {attributes}")
            if parent_run_id in self.runs:
                self.runs[run_id] = self.runs[parent_run_id].create_generation(name=name, attributes=attributes, version=self.version, prompt_slug=prompt_slug)
            else:
                self.runs[run_id] = self.trace.create_generation(name=name, attributes=attributes, version=self.version, prompt_slug=prompt_slug)

        except Exception as e:
            _debug(e)


    def _generate_trace(
        self,
        serialized: Dict[str, Any],
        inputs: Union[Dict[str, Any], List[str], str, None],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        try:
            class_name = self._get_athina_run_name(serialized, **kwargs)
            # Initialise trace by creating a trace if it does not exist
            if self.trace is None:
                trace = Trace(
                    name=self.trace_name if self.trace_name is not None else class_name,
                    attributes=metadata,
                    version=self.version,
                )
                self.trace = trace 
                self.trace_run_id = run_id
        except Exception as e:
            _debug(e)

    def _join_tags_and_metadata(
        self,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if tags is None and metadata is None:
            return None
        elif tags is not None and len(tags) > 0:
            final_dict = {"tags": tags}
            if metadata is not None:
                final_dict.update(metadata)  # Merge metadata into final_dict
            return final_dict
        else:
            return metadata
 
    def _update_run(self, run_id: str, output: Dict, error: Optional[Exception] = None):
        """Update the trace/span with the output of the current run."""
        if self.trace is not None and self.runs[run_id] is not None:
            if error is not None:
                self.runs[run_id].update(status="ERROR",attributes={"status_message": str(error)})
            else:
                self.runs[run_id].update(output=output)

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        # assistant message
        if isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, ToolMessage):
            message_dict = {"role": "tool", "content": message.content}
        elif isinstance(message, FunctionMessage):
            message_dict = {"role": "function", "content": message.content}
        elif isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        else:
            raise ValueError(f"Got unknown type {message}")
        if "name" in message.additional_kwargs:
            message_dict["name"] = message.additional_kwargs["name"]

        if message.additional_kwargs:
            message_dict["additional_kwargs"] = message.additional_kwargs

        return message_dict

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        return [self._convert_message_to_dict(m) for m in messages]


def _extract_raw_response(last_response):
    """Extract the response from the last response of the LLM call."""
    # We return the text of the response if not empty, otherwise the additional_kwargs
    # Additional kwargs contains the response in case of tool usage
    return (
        last_response.text.strip()
        if last_response.text is not None and last_response.text.strip() != ""
        else last_response.message.additional_kwargs
    )

def _flatten_comprehension(matrix):
    return [item for row in matrix for item in row]

def _debug(msg):
    if get_debug():
        print(msg)