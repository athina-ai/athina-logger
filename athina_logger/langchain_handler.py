from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    ChatMessage,
    LLMResult,
)
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.document import Document

from .inference_logger import InferenceLogger
from .api_key import AthinaApiKey
from .util.token_count_helper import get_prompt_tokens_openai_chat_completion, get_completion_tokens_openai_chat_completion, get_token_usage_openai_completion


class CallbackHandler(BaseCallbackHandler, AthinaApiKey):
    """
    Callback handler for the LangChain API.
    """

    def __init__(
        self,
        prompt_slug: str,
        user_query: Optional[str] = None,
        environment: Optional[str] = 'production',
        session_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        customer_user_id: Optional[str] = None,
        external_reference_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the CallbackHandler"""
        if kwargs:
            self.global_context = kwargs
        else:
            self.global_context = None

        self.prompt_slug = prompt_slug
        self.user_query = user_query
        self.environment = environment
        self.session_id = session_id
        self.customer_id = customer_id
        self.customer_user_id = customer_user_id
        self.external_reference_id = external_reference_id
        self.runs: Dict[UUID, Dict[str, Any]] = {}

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            if run_id is None or run_id not in self.runs:
                return
            run_info = self.runs[run_id]
            retrieved_documents_data = ''
            for document in documents:
                page_content = document.page_content
                retrieved_documents_data += page_content + '\n'
            self.global_context['documents'] = retrieved_documents_data
            run_info['retrieved_documents'] = self.global_context
        except Exception as e:
            exception_message = (
                f"Error:\n"
                f"service name: athina-logger\n"
                f"file name: langchain_handler\n"
                f"method name: on_retriever_end\n"
                f"{str(e)}"
            )
            print(exception_message)

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any,
    ) -> Any:
        """
        Log the chat model start
        """
        try:
            for message in messages:
                message_dicts = self._create_message_dicts(message)

            self.runs[run_id] = {
                'is_chat_model': True,
                'prompt_slug': self.prompt_slug,
                'user_query': self.user_query,
                'context': self.global_context,
                'prompt_sent': message_dicts,
                'session_id': self.session_id,
                'customer_id': self.customer_id,
                'customer_user_id': self.customer_user_id,
                'external_reference_id': self.external_reference_id,
                'llm_start_time': datetime.now(timezone.utc),
                'language_model_id': kwargs.get('invocation_params').get('model_name')
            }
        except Exception as e:
            exception_message = (
                f"Error:\n"
                f"service name: athina-logger\n"
                f"file name: langchain_handler\n"
                f"method name: on_chat_model_start\n"
                f"{str(e)}"
            )
            print(exception_message)

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        pass

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            self.runs[run_id] = {
                'is_chat_model': False,
                'prompt_slug': self.prompt_slug,
                'user_query': self.user_query,
                'context': self.global_context,
                'prompt_sent': {'text': ' '.join(prompts)},
                'session_id': self.session_id,
                'customer_id': self.customer_id,
                'customer_user_id': self.customer_user_id,
                'external_reference_id': self.external_reference_id,
                'llm_start_time': datetime.now(timezone.utc),
                'language_model_id': kwargs.get('invocation_params').get('model_name')
            }
        except Exception as e:
            exception_message = (
                f"Error:\n"
                f"service name: athina-logger\n"
                f"file name: langchain_handler\n"
                f"method name: on_llm_start\n"
                f"{str(e)}"
            )
            print(exception_message)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            run_info = self.runs.get(run_id, {})
            if not run_info:
                return
            llm_end_time = datetime.now(timezone.utc)
            run_info['response_time'] = round((
                (llm_end_time - run_info['llm_start_time']).total_seconds())*1000)

            for i in range(len(response.generations)):
                generation = response.generations[i][0]
                prompt_response = generation.text
                llm_output = response.llm_output
                token_usage = self._get_llm_usage(llm_output=llm_output, prompt_sent=run_info['prompt_sent'],
                                                  prompt_response=prompt_response, model=run_info['language_model_id'], is_chat_model=run_info['is_chat_model'])

                run_info['prompt_response'] = prompt_response
                run_info['prompt_tokens'] = token_usage['prompt_tokens']
                run_info['completion_tokens'] = token_usage['completion_tokens']
                run_info['total_tokens'] = token_usage['total_tokens']

                # LOG TO API SERVER
                self._log_llm_response(run_info)
        except Exception as e:
            exception_message = (
                f"Error:\n"
                f"service name: athina-logger\n"
                f"file name: langchain_handler\n"
                f"method name: on_llm_end\n"
                f"{str(e)}"
            )
            print(exception_message)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool starts."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing when agent takes a specific action."""
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool ends."""
        pass

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when tool outputs an error."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing"""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Do nothing"""
        pass

    def _get_llm_usage(self, llm_output: Dict, prompt_sent, prompt_response, model, is_chat_model) -> Dict:
        """
        Fetch prompt tokens, completion tokens and total tokens from llm output
        """
        if llm_output is not None and 'token_usage' in llm_output:
            return {
                'prompt_tokens': llm_output['token_usage']['prompt_tokens'] if 'prompt_tokens' in llm_output['token_usage'] else None,
                'completion_tokens': llm_output['token_usage']['completion_tokens'] if 'completion_tokens' in llm_output['token_usage'] else None,
                'total_tokens': llm_output['token_usage']['total_tokens'] if 'total_tokens' in llm_output['token_usage'] else None,
            }
        else:
            if is_chat_model:
                prompt_tokens = self._get_prompt_tokens_chat_model(
                    messages=prompt_sent, model=model)
                completion_tokens = self._get_completion_tokens_chat_model(
                    response=prompt_response, model=model)
                if prompt_tokens is not None and completion_tokens is not None:
                    total_tokens = prompt_tokens + completion_tokens
                else:
                    total_tokens = None
            else:
                prompt_tokens = self._get_token_usage_completion_model(
                    text=prompt_sent, model=model)
                completion_tokens = self._get_token_usage_completion_model(
                    text=prompt_response, model=model)
                if prompt_tokens is not None and completion_tokens is not None:
                    total_tokens = prompt_tokens + completion_tokens
                else:
                    total_tokens = None
        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens
        }

    def _get_prompt_tokens_chat_model(self, messages: List[Dict[str, Any]], model: str):
        """
        gets the prompt tokens given the messages array
        """
        try:
            tokens = get_prompt_tokens_openai_chat_completion(
                messages=messages, model=model)
            return tokens
        except Exception as e:
            return None

    def _get_completion_tokens_chat_model(self, response: str, model: str):
        """
        gets the completion tokens given the prompt response from the openai chat model completion
        """
        try:
            tokens = get_completion_tokens_openai_chat_completion(
                response=response, model=model)
            return tokens
        except Exception as e:
            return None

    def _get_token_usage_completion_model(self, text: str, model: str):
        """
        gets the token usage given the prompt or prompt response for the openai completion model
        """
        try:
            tokens = get_token_usage_openai_completion(
                text=text, model=model)
            return tokens
        except Exception as e:
            return None

    def _log_llm_response(self, run_info: Dict):
        """
        Logs the LLM response to athina
        """
        try:
            InferenceLogger.log_langchain_llm_response(prompt_slug=run_info['prompt_slug'], prompt_sent=run_info['prompt_sent'],
                                                       prompt_response=run_info['prompt_response'], model=run_info['language_model_id'],
                                                       prompt_tokens=run_info['prompt_tokens'], completion_tokens=run_info[
                                                           'completion_tokens'],
                                                       total_tokens=run_info['total_tokens'], response_time=run_info['response_time'],
                                                       environment=self.environment, context=run_info[
                'context'], user_query=run_info['user_query'],
                customer_id=run_info['customer_id'], session_id=run_info['session_id'],
                customer_user_id=run_info['customer_user_id'], external_reference_id=run_info['external_reference_id'])

        except Exception as e:
            exception_message = (
                f"Error:\n"
                f"service name: athina-logger\n"
                f"file name: langchain_handler\n"
                f"method name: _log_llm_response\n"
                f"{str(e)}"
            )
            print(exception_message)

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        if isinstance(message, HumanMessage):
            message_dict = {'role': 'user', 'content': message.content}
        elif isinstance(message, AIMessage):
            message_dict = {'role': 'assistant', 'content': message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {'role': 'system', 'content': message.content}
        elif isinstance(message, ChatMessage):
            message_dict = {'role': message.role, 'content': message.content}
        else:
            raise ValueError(f'Got unknown type {message}')
        if 'name' in message.additional_kwargs:
            message_dict['name'] = message.additional_kwargs['name']
        return message_dict

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts
