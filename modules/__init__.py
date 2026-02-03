# modules/__init__.py
from abc import ABC, abstractmethod

class Evaluator(ABC):
    """
    Abstract base class for all utterance evaluators.
    Each subclass should implement an `evaluate(text: str)` method
    """

    def __init__(self, name: str):
        self.name = name

    def evaluate_utterance(self, text: str) -> dict:
        """Evaluate a single utterance. Override if applicable."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support utterance-level evaluation"
        )
    
    def evaluate_utterance_pair(self, text1: str, text2: str) -> dict:
        """Evaluate a pair of utterances. Override if applicable."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support pair-level evaluation"
        )
    
    def evaluate_conversation(self, conversation: list[str]) -> dict:
        """Evaluate a conversation. Override if applicable."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support conversation-level evaluation"
        )

    def __repr__(self):
        return f"<Evaluator {self.name}>"