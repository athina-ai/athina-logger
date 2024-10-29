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
