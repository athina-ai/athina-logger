from typing import Dict, List, Any
import tiktoken
from ..constants import OPENAI_MODEL_ENCODINGS

# source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb


def get_prompt_tokens_openai_chat_completion(prompt: List[Dict[str, Any]], language_model_id: str):
    """
    gets the prompt tokens given the prompt for the openai chat model completion
    """
    if prompt is None:
        raise ValueError('prompt is None')

    if language_model_id in {
        'gpt-3.5-turbo-0613',
        'gpt-3.5-turbo-16k-0613',
        'gpt-4-0613',
        'gpt-4-32k-0613',
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif 'gpt-3.5-turbo' in language_model_id:
        return get_prompt_tokens_openai_chat_completion(prompt=prompt, language_model_id='gpt-3.5-turbo-0613')
    elif 'gpt-4' in language_model_id:
        return get_prompt_tokens_openai_chat_completion(prompt=prompt, language_model_id='gpt-4-0613')
    else:
        raise ValueError(
            f'Language model {language_model_id} is not supported')

    try:
        encoding = tiktoken.get_encoding(OPENAI_MODEL_ENCODINGS[language_model_id])
    except KeyError:
        return None

    try:
        num_tokens = 0
        for message in prompt:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == 'name':
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens
    except Exception as e:
        raise e


def get_completion_tokens_openai_chat_completion(response: str, language_model_id: str):
    """
    gets the completion tokens given the prompt response from the openai chat model completion
    """
    if response is None:
        raise ValueError('response is None')
    try:
        if language_model_id in {
            'gpt-3.5-turbo-0613',
            'gpt-3.5-turbo-16k-0613',
            'gpt-4-0613',
            'gpt-4-32k-0613',
        }:
            encoding = tiktoken.get_encoding(OPENAI_MODEL_ENCODINGS[language_model_id])
        elif 'gpt-3.5-turbo' in language_model_id:
            return get_completion_tokens_openai_chat_completion(response=response, language_model_id='gpt-3.5-turbo-0613')
        elif 'gpt-4' in language_model_id:
            return get_completion_tokens_openai_chat_completion(response=response, language_model_id='gpt-4-0613')
        else:
            raise ValueError(
                f'Language model {language_model_id} is not supported')
    except KeyError:
        return None

    try:
        tokens = len(encoding.encode(response))
        return tokens
    except Exception as e:
        raise e


def get_token_usage_openai_completion(text: str, language_model_id: str):
    """
    gets the token usage given the text and language_model_id for openai completion
    """
    if text is None:
        raise ValueError('text is None')
    try:
        encoding = tiktoken.get_encoding(OPENAI_MODEL_ENCODINGS[language_model_id])
    except KeyError:
        return None

    try:
        tokens = len(encoding.encode(text))
        return tokens
    except Exception as e:
        raise e
