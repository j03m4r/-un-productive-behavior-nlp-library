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