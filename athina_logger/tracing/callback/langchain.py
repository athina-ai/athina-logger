import logging

from athina_logger.api_key import AthinaApiKey
 
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID, uuid4
from langchain.callbacks.base import BaseCallbackHandler as LangchainBaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
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

class LangchainCallbackHandler(
    LangchainBaseCallbackHandler, AthinaApiKey
):
    log = logging.getLogger("athina")
    next_span_id: Optional[str] = None

    def __init__(
        self,
        trace_name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None: 
        self.version = version
        self.trace_name = trace_name
        self.runs = {}
        self.trace = None 
        self.root_span = None

    def setNextSpan(self, id: str):
        self.next_span_id = id

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.log.debug(
            f"on llm new token: run_id: {run_id} parent_run_id: {parent_run_id}"
        )

    def get_athina_run_name(self, serialized: Dict[str, Any], **kwargs: Any) -> str:
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
            return kwargs["name"]

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
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(
                f"on chain start: run_id: {run_id} parent_run_id: {parent_run_id}, name {serialized.get('name', serialized.get('id', ['<unknown>'])[-1])}"
            )
            self.__generate_trace_and_parent(
                serialized=serialized,
                inputs=inputs,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
                metadata=metadata,
                version=self.version,
                **kwargs,
            )

            content = {
                "id": self.next_span_id,
                "trace_id": self.trace.id,
                "name": self.get_athina_run_name(serialized, **kwargs),
                "metadata": self.__join_tags_and_metadata(tags, metadata),
                "input": inputs,
                "version": self.version,
            }

            if parent_run_id is None:
                if self.root_span is None:
                    self.runs[run_id] = self.trace.add_span(name=content["name"], attributes=content, version=self.version)
                else:
                    self.runs[run_id] = self.root_span.add_span(name=content["name"], attributes=content, version=self.version)
            if parent_run_id is not None:
                self.runs[run_id] = self.runs[parent_run_id].add_span(name=content["name"], attributes=content, version=self.version)

        except Exception as e:
            self.log.exception(e)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(
                f"on chain end: run_id: {run_id} parent_run_id: {parent_run_id}"
            )

            if run_id not in self.runs:
                raise Exception("run not found")
            
            self._update_trace(run_id, parent_run_id, outputs)
            self.runs[run_id] = self.runs[run_id].end()
        except Exception as e:
            self.log.exception(e)

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
            self.log.debug(
                f"on chain error: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            self._update_trace(run_id, parent_run_id, None, error)
            self.runs[run_id] = self.runs[run_id].end()

        except Exception as e:
            self.log.exception(e)

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
            self.log.debug(
                f"on agent action: run_id: {run_id} parent_run_id: {parent_run_id}"
            )

            if run_id not in self.runs:
                raise Exception("run not found")

            self._update_trace(run_id, parent_run_id, action, None)

            self.runs[run_id] = self.runs[run_id].end()

        except Exception as e:
            self.log.exception(e)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(
                f"on agent finish: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            if run_id not in self.runs:
                raise Exception("run not found")

            self._update_trace(run_id, parent_run_id, finish, None)
            self.runs[run_id] = self.runs[run_id].end()

        except Exception as e:
            self.log.exception(e)

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
            self.log.debug(
                f"on chat model start: run_id: {run_id} parent_run_id: {parent_run_id}"
            )

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
            self.log.exception(e)

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
            self.log.debug(
                f"on retriever start: run_id: {run_id} parent_run_id: {parent_run_id}"
            )

            if parent_run_id is None or parent_run_id not in self.runs:
                raise Exception("parent run not found")

            self.runs[run_id] = self.runs[parent_run_id].span(
                id=self.next_span_id,
                name=self.get_athina_run_name(serialized, **kwargs),
                input=query,
                metadata=self.__join_tags_and_metadata(tags, metadata),
                version=self.version,
            )
            self.next_span_id = None
        except Exception as e:
            self.log.exception(e)

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(
                f"on retriever end: run_id: {run_id} parent_run_id: {parent_run_id}"
            )

            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")

            self._update_trace(run_id, parent_run_id, documents, None)
            self.runs[run_id] = self.runs[run_id].end()

        except Exception as e:
            self.log.exception(e)

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
            self.log.debug(
                f"on retriever error: run_id: {run_id} parent_run_id: {parent_run_id}"
            )

            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")
            self._update_trace(run_id, parent_run_id, None, error)
            self.runs[run_id] = self.runs[run_id].end()

        except Exception as e:
            self.log.exception(e)

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
            self.log.debug(
                f"on tool start: run_id: {run_id} parent_run_id: {parent_run_id}"
            )

            if parent_run_id is None or parent_run_id not in self.runs:
                raise Exception("parent run not found")
            meta = self.__join_tags_and_metadata(tags, metadata)

            meta.update(
                {key: value for key, value in kwargs.items() if value is not None}
            )

            self.runs[run_id] = self.runs[parent_run_id].span(
                id=self.next_span_id,
                name=self.get_athina_run_name(serialized, **kwargs),
                input=input_str,
                metadata=meta,
                version=self.version,
            )
            self.next_span_id = None
        except Exception as e:
            self.log.exception(e)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(
                f"on tool end: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")
            self._update_trace(run_id, parent_run_id, output, None)
            self.runs[run_id] = self.runs[run_id].end(
                output=output, version=self.version
            )

        except Exception as e:
            self.log.exception(e)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(
                f"on tool error: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            if run_id is None or run_id not in self.runs:
                raise Exception("run not found")
            self._update_trace(run_id, parent_run_id, None, error)
            self.runs[run_id] = self.runs[run_id].end()

        except Exception as e:
            self.log.exception(e)

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
            self.log.debug(
                f"on llm start: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
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
            self.log.exception(e)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(
                f"on llm end: run_id: {run_id} parent_run_id: {parent_run_id} response: {response} kwargs: {kwargs}"
            )
            if run_id not in self.runs:
                raise Exception("Run not found, see docs what to do in this case.")
            else:
                generation = response.generations[-1][-1]
                extracted_response = (
                    self._convert_message_to_dict(generation.message)
                    if isinstance(generation, ChatGeneration)
                    else _extract_raw_esponse(generation)
                )
                llm_usage = (
                    None
                    if response.llm_output is None
                    or not response.llm_output["token_usage"]
                    else response.llm_output["token_usage"]
                )
                # self._update_trace(run_id, parent_run_id, extracted_response, None)
                self.runs[run_id] = self.runs[run_id].end()
                # print("run_id", run_id)
                # print("parent_run_id", parent_run_id)
                # print(self.trace)
                self.trace.end()
        except Exception as e:
            self.log.exception(e)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.log.debug(
                f"on llm error: run_id: {run_id} parent_run_id: {parent_run_id}"
            )
            self._update_trace(run_id, parent_run_id, None, error)
            self.runs[run_id] = self.runs[run_id].end()

        except Exception as e:
            self.log.exception(e)

    def _on_llm_action(
        self,
        serialized: Dict[str, Any],
        run_id: UUID,
        prompts: List[str],
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        try:
            self.__generate_trace_and_parent(
                serialized,
                inputs=prompts[0] if len(prompts) == 1 else prompts,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
                metadata=metadata,
                version=self.version,
                kwargs=kwargs,
            )
            model_name = kwargs.get('invocation_params').get('model_name')
            content = {
                "name": self.get_athina_run_name(serialized, **kwargs),
                "input": prompts,
                "metadata": self.__join_tags_and_metadata(tags, metadata),
                "model": model_name,
                "model_parameters": {
                    key: value
                    for key, value in {
                        "temperature": kwargs["invocation_params"].get("temperature"),
                        "max_tokens": kwargs["invocation_params"].get("max_tokens"),
                        "top_p": kwargs["invocation_params"].get("top_p"),
                        "frequency_penalty": kwargs["invocation_params"].get(
                            "frequency_penalty"
                        ),
                        "presence_penalty": kwargs["invocation_params"].get(
                            "presence_penalty"
                        ),
                        "request_timeout": kwargs["invocation_params"].get(
                            "request_timeout"
                        ),
                    }.items()
                    if value is not None
                },
                "version": self.version,
            }

            if parent_run_id in self.runs:
                self.runs[run_id] = self.runs[parent_run_id].add_generation(name=content["name"], attributes=content, version=self.version)
            elif self.root_span is not None and parent_run_id is None:
                self.runs[run_id] = self.root_span.add_generation(name=content["name"], attributes=content, version=self.version)
            else:
                self.runs[run_id] = self.trace.add_generation(name=content["name"], attributes=content, version=self.version)

        except Exception as e:
            self.log.exception(e)


    def __generate_trace_and_parent(
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
            class_name = self.get_athina_run_name(serialized, **kwargs)

            # on a new invocation, and not user provided root, we want to initialise a new trace
            # parent_run_id is None when we are at the root of a langchain execution
            if (
                self.trace is not None
                and parent_run_id is None
            ):
                self.trace = None
                self.runs = {}

            # if we are at a root, it means we do not have a root provided by a user.
            # Initialise it by creating a trace and root span.
            if self.trace is None:
                trace = Trace(
                    name=self.trace_name if self.trace_name is not None else class_name,
                    version=self.version,
                )

                self.trace = trace

                if parent_run_id is not None and parent_run_id in self.runs:
                    self.runs[run_id] = self.trace.add_span(
                        name=class_name,
                        input=inputs,
                        version=self.version,
                    )

                return

        except Exception as e:
            self.log.exception(e)

    def __join_tags_and_metadata(
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
 
    def _update_trace(self, run_id: str, parent_run_id: Optional[str], output: any, error: Optional[Exception] = None):
        """Update the trace with the output of the current run. Called at every finish callback event."""
        if self.trace is not None:
            if error is not None:
                self.trace.update(status="ERROR",attributes={"status_message": str(error)}),
            else:
                self.trace.update(output=output)

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


def _extract_raw_esponse(last_response):
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
