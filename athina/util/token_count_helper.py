from typing import Dict, List, Any
import tiktoken

# source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb


def get_prompt_tokens_openai_chat_completion(messages: List[Dict[str, Any]], model: str):
    """
    gets the prompt tokens given the messages array
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding('cl100k_base')
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
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == 'name':
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def get_completion_tokens_openai_chat_completion(response: str, model: str):
    """
    gets the completion tokens given the prompt response from the openai chat model completion
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding('cl100k_base')

    try:
        tokens = len(encoding.encode(response))
        return tokens
    except Exception as e:
        raise e
