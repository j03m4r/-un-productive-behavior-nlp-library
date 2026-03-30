from modules import Evaluator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

class SentimentEvaluator(Evaluator):
    def __init__(self):
        super().__init__(name="Sentiment")
        self.analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None, tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")

    def evaluate_utterance(self, text: str) -> dict:
        score = self.analyzer(text)
        return score

    def evaluate_conversation(self, conversation):
        res = {
            "aggregate": {

            },
            "utterances": []
        }

        for utterance in conversation:
            score = self.evaluate_utterance(utterance)
            res["utterances"].append(score)
            for key in score.keys():
                res['aggregate'][key] = res['aggregate'].get(key, 0) + score[key] 

        for key in res['aggregate'].keys():
            res['aggregate'][key] /= len(res['utterances'])

        return res