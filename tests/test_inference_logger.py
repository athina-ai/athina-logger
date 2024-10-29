import os

import pytest

from athina_logger.api_key import AthinaApiKey
from athina_logger.inference_logger import InferenceLogger


@pytest.mark.parametrize(
    "language_model_id, prompt, response, user_query, context, prompt_tokens, "
    "completion_tokens, total_tokens, cost, response_time, prompt_slug, environment, "
    "customer_id, customer_user_id, session_id, expected_response, tools, tool_calls, "
    "functions, function_call_response, external_reference_id, custom_attributes",
    [
        (
                "gpt-4",
                [
                    {
                        "role": "system",
                        "content": "Answer the following question using the information provided.\n ### INFORMATION ### Neil Armstrong landed on the moon in 1969.\n ### QUERY ###"
                    },
                    {
                        "role": "user",
                        "content": "Which spaceship was first to land on the moon?"
                    }
                ],
                "The Apollo 11 was the first spaceship to land on the moon.",
                "Which spaceship was first to land on the moon?",
                {"information": ["Neil Armstrong landed on the moon in 1969."]},
                22,
                9,
                31,
                0.002,
                150,
                "qa_chatbot_response",
                "development",
                "xyz-123",
                "user-456",
                "session-789",
                "The Apollo 11 was the first spaceship to land on the moon.",
                [],
                None,
                [],
                None,
                "ref-101112",
                {"company": "OpenAI", "links": ["https://openai.com"]},
        ),
    ]
)
def test_log_inference_no_model_options(
        language_model_id, prompt, response, user_query, context, prompt_tokens,
        completion_tokens, total_tokens, cost, response_time, prompt_slug, environment,
        customer_id, customer_user_id, session_id, expected_response, tools, tool_calls,
        functions, function_call_response, external_reference_id, custom_attributes
):
    athina_api_key = os.getenv("ATHINA_API_KEY")

    AthinaApiKey.set_api_key(api_key=athina_api_key)

    InferenceLogger.log_inference(
        language_model_id=language_model_id,
        prompt=prompt,
        response=response,
        user_query=user_query,
        context=context,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        response_time=response_time,
        prompt_slug=prompt_slug,
        environment=environment,
        customer_id=customer_id,
        customer_user_id=customer_user_id,
        session_id=session_id,
        expected_response=expected_response,
        tools=tools,
        tool_calls=tool_calls,
        functions=functions,
        function_call_response=function_call_response,
        external_reference_id=external_reference_id,
        custom_attributes=custom_attributes,
    )


@pytest.mark.parametrize(
    "language_model_id, prompt, response, user_query, context, prompt_tokens, "
    "completion_tokens, total_tokens, cost, response_time, prompt_slug, environment, "
    "customer_id, customer_user_id, session_id, expected_response, tools, tool_calls, "
    "functions, function_call_response, external_reference_id, custom_attributes, model_options, ",
    [
        (
                "gpt-4",
                [
                    {
                        "role": "system",
                        "content": "Answer the following question using the information provided.\n ### INFORMATION ### Neil Armstrong landed on the moon in 1969.\n ### QUERY ###"
                    },
                    {
                        "role": "user",
                        "content": "Which spaceship was first to land on the moon?"
                    }
                ],
                "The Apollo 11 was the first spaceship to land on the moon.",
                "Which spaceship was first to land on the moon?",
                {"information": ["Neil Armstrong landed on the moon in 1969."]},
                22,
                9,
                31,
                0.002,
                150,
                "qa_chatbot_response",
                "development",
                "xyz-123",
                "user-456",
                "session-789",
                "The Apollo 11 was the first spaceship to land on the moon.",
                [],
                None,
                [],
                None,
                "ref-101112",
                {"company": "OpenAI", "links": ["https://openai.com"]},
                None
        ),
    ]
)
def test_log_inference_with_model_options_none(
        language_model_id, prompt, response, user_query, context, prompt_tokens,
        completion_tokens, total_tokens, cost, response_time, prompt_slug, environment,
        customer_id, customer_user_id, session_id, expected_response, tools, tool_calls,
        functions, function_call_response, external_reference_id, custom_attributes, model_options
):
    athina_api_key = os.getenv("ATHINA_API_KEY")

    AthinaApiKey.set_api_key(api_key=athina_api_key)

    InferenceLogger.log_inference(
        language_model_id=language_model_id,
        prompt=prompt,
        response=response,
        user_query=user_query,
        context=context,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        response_time=response_time,
        prompt_slug=prompt_slug,
        environment=environment,
        customer_id=customer_id,
        customer_user_id=customer_user_id,
        session_id=session_id,
        expected_response=expected_response,
        tools=tools,
        tool_calls=tool_calls,
        functions=functions,
        function_call_response=function_call_response,
        external_reference_id=external_reference_id,
        custom_attributes=custom_attributes,
        model_options=model_options,
    )


## these parameters are shared across the model_options tests
default_params = {
    "language_model_id": "gpt-4",
    "prompt": [
        {
            "role": "system",
            "content": "Answer the following question using the information provided.\n ### INFORMATION ### Neil Armstrong landed on the moon in 1969.\n ### QUERY ###"
        },
        {
            "role": "user",
            "content": "Which spaceship was first to land on the moon?"
        }
    ],
    "response": "The Apollo 11 was the first spaceship to land on the moon.",
    "user_query": "Which spaceship was first to land on the moon?",
    "context": {"information": ["Neil Armstrong landed on the moon in 1969."]},
    "prompt_tokens": 22,
    "completion_tokens": 9,
    "total_tokens": 31,
    "cost": 0.002,
    "response_time": 150,
    "prompt_slug": "qa_chatbot_response",
    "environment": "development",
    "customer_id": "xyz-123",
    "customer_user_id": "user-456",
    "session_id": "session-789",
    "expected_response": "The Apollo 11 was the first spaceship to land on the moon.",
    "tools": [],
    "tool_calls": None,
    "functions": [],
    "function_call_response": None,
    "external_reference_id": "ref-101112",
    "custom_attributes": {"company": "OpenAI", "links": ["https://openai.com"]},
}


@pytest.mark.parametrize("model_options", [
    # no options provided
    {},

    # individual fields
    {"temperature": 0.7},
    {"max_completion_tokens": 150},
    {"stop": "complete"},
    {"stop": ["complete", "end"]},
    {"top_p": 0.9},
    {"extra_options": {"custom_option": "custom_value"}},

    # combinations of fields
    {"temperature": 0.7, "max_completion_tokens": 150},
    {"temperature": 0.7, "stop": "complete"},
    {"temperature": 0.7, "stop": ["complete", "end"]},
    {"temperature": 0.7, "top_p": 0.9},
    {"temperature": 0.7, "extra_options": {"custom_option": "custom_value"}},

    {"max_completion_tokens": 150, "stop": "complete"},
    {"max_completion_tokens": 150, "stop": ["complete", "end"]},
    {"max_completion_tokens": 150, "top_p": 0.9},
    {"max_completion_tokens": 150, "extra_options": {"another_option": 123}},

    {"stop": "complete", "top_p": 0.9},
    {"stop": ["complete", "end"], "top_p": 0.9},
    {"stop": "complete", "extra_options": {"flag": True}},
    {"stop": ["complete", "end"], "extra_options": {"flag": True}},

    {"top_p": 0.9, "extra_options": {"threshold": 0.5}},

    # all fields filled
    {
        "temperature": 0.7,
        "max_completion_tokens": 150,
        "stop": ["complete", "end"],
        "top_p": 0.9,
        "extra_options": {"custom_key": "custom_value"}
    },
])
def test_log_inference_with_model_options_not_none(model_options):
    athina_api_key = os.getenv("ATHINA_API_KEY")
    AthinaApiKey.set_api_key(api_key=athina_api_key)

    test_params = {**default_params, "model_options": model_options}

    InferenceLogger.log_inference(**test_params)
    pass
