# Athina Server Base Url
API_BASE_URL = 'https://log.athina.ai'
LOG_OPENAI_CHAT_COMPLETION_URL = f'{API_BASE_URL}/api/v1/log/prompt/openai-chat'
LOG_OPENAI_COMPLETION_URL = f'{API_BASE_URL}/api/v1/log/prompt/openai-completion'

OPENAI_MODEL_ENCODINGS = {
    'gpt-3.5-turbo-0613': 'cl100k_base',
    'gpt-3.5-turbo-16k-0613': 'cl100k_base',
    'gpt-4-0613': 'cl100k_base',
    'gpt-4-32k-0613': 'cl100k_base',
    'text-davinci-003': 'p50k_base',
}
