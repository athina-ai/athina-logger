# Athina Server Base Url
API_BASE_URL = 'https://log.athina.ai'
# API_BASE_URL = 'https://api.staging.athina.ai'
# API_BASE_URL = 'http://localhost:9000'
LOG_INFERENCE_URL = f'{API_BASE_URL}/api/v1/log/inference'

OPENAI_MODEL_ENCODINGS = {
    'gpt-3.5-turbo-0613': 'cl100k_base',
    'gpt-3.5-turbo-16k-0613': 'cl100k_base',
    'gpt-4-0613': 'cl100k_base',
    'gpt-4-32k-0613': 'cl100k_base',
    'text-davinci-003': 'p50k_base',
}
