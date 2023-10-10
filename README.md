## Overview
Athina Logger is a python SDK to log LLM inferences to Athina API Server. 


## Installation

```bash
pip install athina-logger
```

## Usage
```python
import openai
from athina_logger.inference_logger import InferenceLogger
from athina_logger.log_stream_inference.openai_chat_completion_stream import LogOpenAiChatCompletionStreamInference
from athina_logger.log_stream_inference.openai_completion_stream import LogOpenAiCompletionStreamInference
from athina_logger.api_key import AthinaApiKey
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from athina_logger.langchain_handler import CallbackHandler
from athina_logger.feedback import user_feedback
import sseclient
import requests
import json
from athina_logger.exception.custom_exception import CustomException

openai.api_key = 'OPENAI_API_KEY'

AthinaApiKey.set_api_key('ATHINA_API_KEY')


def performRequestWithStreaming():
    reqUrl = 'https://api.openai.com/v1/completions'
    reqHeaders = {
        'Accept': 'text/event-stream',
        'Authorization': 'Bearer ' + 'OPENAI_API_KEY'
    }
    reqBody = {
        "model": "text-davinci-003",
        "prompt": "What is Python?",
        "stream": True,
    }
    request = requests.post(reqUrl, stream=True,
                            headers=reqHeaders, json=reqBody)
    logger = LogOpenAiCompletionStreamInference(
        prompt_slug="test",
        prompt="What is Python?",
        model="text-davinci-003",
    )
    client = sseclient.SSEClient(request)
    for event in client.events():
        if event.data != '[DONE]':
            logger.collect_stream_inference_by_chunk(json.loads(event.data))

    logger.log_stream_inference()


def user_feedback_test():
    try:
        user_feedback.UserFeedback.log_user_feedback(
            external_reference_id="abc",
            user_feedback=1,
            user_feedback_comment="test"
        )
    except Exception as e:
        if isinstance(e, CustomException):
            print(e.status_code)
            print(e.message)
        else:
            print(e)


def openai_chat_completion(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        InferenceLogger.log_open_ai_chat_response(
            prompt_slug=None,
            messages=[{"role": "user", "content": prompt}],
            model="gpt-3.5-turbo",
            completion=response,
            prompt_response="Hello, How may I assist you?",
            external_reference_id="abc"
        )
    except Exception as e:
        if isinstance(e, CustomException):
            print(e.status_code)
            print(e.message)
        else:
            print(e)


def openai_completion(prompt):
    response = openai.Completion.create(
        model="text-davinci-003", prompt=prompt)

    InferenceLogger.log_open_ai_completion_response(
        prompt_slug="test",
        prompt=prompt,
        model="text-davinci-003",
        completion=response,
        prompt_response="Hello, How may I assist you?",
        external_reference_id="abc"
    )


def generic_logging():
    InferenceLogger.log_generic_response(
        prompt_slug="test",
        prompt="Hello, I'm a human.",
        model="text-davinci-003",
        prompt_response="Hello, How may I assist you?",
        prompt_tokens=10,
        completion_tokens=10,
        external_reference_id="abc"
    )


def stream_openai_chat_response(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    logger = LogOpenAiChatCompletionStreamInference(
        prompt_slug="test",
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",
    )
    for chunk in response:
        logger.collect_stream_inference_by_chunk(chunk)
    logger.log_stream_inference()


def stream_openai_response(prompt):
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        max_tokens=1000,
        stream=True,
    )
    logger = LogOpenAiCompletionStreamInference(
        prompt_slug="test",
        prompt=prompt,
        model="text-davinci-003",
    )
    logger.collect_stream_inference(response)
    logger.log_stream_inference()


def test_langchain_streaming():
    template = '''You are a helpful assistant who generates comma separated lists.
    A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
    ONLY return a comma separated list, and nothing more.'''
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = '{text}'
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])
    system_template = '''You are a helpful assistant who generates lines about a particular topic'''
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template)
    template = '''Write few lines on the following topic: {text}
                Your response:'''
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, HumanMessagePromptTemplate.from_template(template)])
    handler = CallbackHandler(prompt_slug='test')

    # without streaming
    chain1 = LLMChain(
        llm=ChatOpenAI(
            openai_api_key='OPENAI_API_KEY', callbacks=[handler]),
        prompt=chat_prompt,
    )
    # with streaming
    chain2 = LLMChain(
        llm=ChatOpenAI(
            openai_api_key='OPENAI_API_KEY', streaming=True, callbacks=[handler]),
        prompt=chat_prompt,
    )
    print(chain1.run('India'))
    print(chain2.run('India'))

```

## Contact 

Please feel free to reach out to akshat@athina.ai or shiv@athina.ai for more information.
