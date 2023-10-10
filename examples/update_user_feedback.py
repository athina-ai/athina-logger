from athina_logger.api_key import AthinaApiKey
from athina_logger.feedback import user_feedback
from athina_logger.exception.custom_exception import CustomException

AthinaApiKey.set_api_key('YOUR_ATHINA_API_KEY')


def user_feedback_test():
    try:
        user_feedback.UserFeedback.log_user_feedback(
            # required; should be unique across all logged prompt runs
            external_reference_id="abc",
            # required; 1: positive, -1: negative
            user_feedback=1,
            # optional
            user_feedback_comment="test"
        )
    except Exception as e:
        if isinstance(e, CustomException):
            print(e.status_code)
            print(e.message)
        else:
            print(e)
