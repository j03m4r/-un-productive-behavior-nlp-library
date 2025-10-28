from modules import Evaluator
from googleapiclient import discovery
import os

class ToxicityEvaluator(Evaluator):
    def __init__(self):
        super().__init__(name="Toxicity")
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=os.getenv("PERSPECTIVE_API_KEY"),
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False
        )

    def evaluate_utterance(self, text: str) -> dict:
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}}
        }
        response = self.client.comments().analyze(body=analyze_request).execute()
        score = float(response["attributeScores"]["TOXICITY"]["summaryScore"]["value"])
        label = "highly-toxic" if score > .8 else "toxic" if score > .5 else "non-toxic" # Threshold values from 'Conversations Gone Alright'

        return {
            "score": round(score, 3),
            "label": label
        }

    def evaluate_conversation(self, conversation: list[str]) -> dict:
        res = {
            "labels": {
                "non-toxic": 0, 
                "toxic": 0,
                "highly-toxic": 0
            },
            "utterances": []
        }

        for text in conversation:
            _res = self.evaluate_utterance(text)
            res["labels"][_res["label"]] += 1
            res["utterances"].append(_res)

        for label in ["non-toxic", "toxic", "highly-toxic"]:
            res["labels"][label] /= len(conversation)

        return res