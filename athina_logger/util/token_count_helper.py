from typing import Dict, List, Any
import tiktoken
from ..constants import OPENAI_MODEL_ENCODINGS

# source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb


def get_prompt_tokens_openai_chat_completion(messages: List[Dict[str, Any]], model: str):
    """
    gets the prompt tokens given the messages array
    """
    if messages is None:
        raise ValueError('messages is None')

    if model in {
        'gpt-3.5-turbo-0613',
        'gpt-3.5-turbo-16k-0613',
        'gpt-4-0613',
        'gpt-4-32k-0613',
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif 'gpt-3.5-turbo' in model:
        return get_prompt_tokens_openai_chat_completion(messages, model='gpt-3.5-turbo-0613')
    elif 'gpt-4' in model:
        return get_prompt_tokens_openai_chat_completion(messages, model='gpt-4-0613')
    else:
        raise ValueError(
            f'Language model {model} is not supported')

    try:
        encoding = tiktoken.get_encoding(OPENAI_MODEL_ENCODINGS[model])
    except KeyError:
        return None

    try:
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == 'name':
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens
    except Exception as e:
        raise e


def get_completion_tokens_openai_chat_completion(response: str, model: str):
    """
    gets the completion tokens given the prompt response from the openai chat model completion
    """
    if response is None:
        raise ValueError('response is None')
    try:
        if model in {
            'gpt-3.5-turbo-0613',
            'gpt-3.5-turbo-16k-0613',
            'gpt-4-0613',
            'gpt-4-32k-0613',
        }:
            encoding = tiktoken.get_encoding(OPENAI_MODEL_ENCODINGS[model])
        elif 'gpt-3.5-turbo' in model:
            return get_completion_tokens_openai_chat_completion(response, model='gpt-3.5-turbo-0613')
        elif 'gpt-4' in model:
            return get_completion_tokens_openai_chat_completion(response, model='gpt-4-0613')
        else:
            raise ValueError(
                f'Language model {model} is not supported')
    except KeyError:
        return None

    try:
        tokens = len(encoding.encode(response))
        return tokens
    except Exception as e:
        raise e


def get_token_usage_openai_completion(text: str, model: str):
    """
    gets the token usage given the text and model for openai completion
    """
    if text is None:
        raise ValueError('text is None')
    try:
        encoding = tiktoken.get_encoding(OPENAI_MODEL_ENCODINGS[model])
    except KeyError:
        return None

    try:
        tokens = len(encoding.encode(text))
        return tokens
    except Exception as e:
        raise e
