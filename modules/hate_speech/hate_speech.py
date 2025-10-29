from transformers import BertTokenizer, BertForSequenceClassification
import torch
from modules import Evaluator
from pathlib import Path

class HateSpeechEvaluator(Evaluator):
    MODEL_PATH = Path(__file__).resolve().parent / "HateBERT_hateval"

    def __init__(self):
        super().__init__(name="Hate Speech")
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_PATH)
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_PATH)
        self.model.eval()

    def evaluate_utterance(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        score = probs[0, 1].item()
        label = "hate" if score > 0.5 else "non-hate"
        return {
            "score": round(score, 3),
            "label": label
        }

    def evaluate_conversation(self, conversation: list[str]):
        res = {
            "aggregate": {
                "non-hate": 0, 
                "hate": 0,
            },
            "utterances": []
        }

        for text in conversation:
            _res = self.evaluate_utterance(text)
            res["aggregate"][_res["label"]] += 1
            res["utterances"].append(_res)

        for label in ["non-hate", "hate"]:
            res["aggregate"][label] /= len(conversation)

        return res