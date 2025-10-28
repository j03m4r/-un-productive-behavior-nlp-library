from modules import Evaluator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentEvaluator(Evaluator):
    def __init__(self):
        super().__init__(name="Sentiment")
        self.analyzer = SentimentIntensityAnalyzer()

    def evaluate_utterance(self, text: str) -> dict:
        score = self.analyzer.polarity_scores(text)
        return score

    def evaluate_conversation(self, conversation):
        scores = []
        for utterance in conversation:
            scores.append(self.evaluate_utterance(utterance))
        return scores