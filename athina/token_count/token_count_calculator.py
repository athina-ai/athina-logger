from abc import ABC, abstractmethod


class TokenCountCalculator(ABC):
    """
    abstract class for token count calculator.
    """

    def __init__(self) -> None:
        """
        constructor for token count calculator
        """
        pass

    @abstractmethod
    def count_tokens(self, tokens):
        """
        counts the tokens
        """
        pass
