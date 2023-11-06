# Athina Server Base Url
API_BASE_URL = 'https://log.athina.ai'
LOG_OPENAI_CHAT_COMPLETION_URL = f'{API_BASE_URL}/api/v1/log/prompt/openai-chat'
LOG_OPENAI_COMPLETION_URL = f'{API_BASE_URL}/api/v1/log/prompt/openai-completion'

LLM_MODELS_SUPPORTED = [
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-16k-0613',
    'gpt-4',
    'gpt-4-0613',
    'gpt-4-32k',
    'gpt-4-32k-0613',
    'meta-llama/Llama-2-13b',
    'meta-llama/Llama-2-13b-chat',
    'meta-llama/Llama-2-13b-chat-hf',
    'meta-llama/Llama-2-13b-hf',
    'meta-llama/Llama-2-70b',
    'meta-llama/Llama-2-70b-chat',
    'meta-llama/Llama-2-70b-chat-hf',
    'meta-llama/Llama-2-70b-hf',
    'meta-llama/Llama-2-7b',
    'meta-llama/Llama-2-7b-chat',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-7b-hf',
    'claude-2',
    'text-davinci-003'
]

OPENAI_MODEL_ENCODINGS = {
    'gpt-3.5-turbo-0613': 'cl100k_base',
    'gpt-3.5-turbo-16k-0613': 'cl100k_base',
    'gpt-4-0613': 'cl100k_base',
    'gpt-4-32k-0613': 'cl100k_base',
    'text-davinci-003': 'p50k_base',
}
