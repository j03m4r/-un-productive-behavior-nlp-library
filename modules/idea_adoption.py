from modules import Evaluator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

class IdeaAdoptionEvaluator(Evaluator):
    def __init__(self):
        super().__init__(name="Idea Adoption")
        self.analyzer = SentimentIntensityAnalyzer()
        self.p1_ideas = {} # (non-stopword) noun, proper noun, verb, adjective -> sentiment when proposing "idea"
        self.p1_adopted_ideas = set()
        self.p2_ideas = {}
        self.p2_adopted_ideas = set()

    def evaluate_conversation(self, conversation: list[str]) -> dict:
        """Captures how many 'ideas' (non-stopword nouns, proper nouns, verbs, and adjectives) that 
        were first proposed by one participant and subsequently adopted by the other participant"""
        nlp = spacy.load("en_core_web_sm")
        candidate_idea_pos_lst = ["VERB", "NOUN", "PROPN", "ADJ"]
        for idx, utterance in enumerate(conversation):
            utterance = utterance.lower()
            doc = nlp(utterance)
            for token in doc:
                if not token.is_stop and token.pos_ in candidate_idea_pos_lst:
                    if idx % 2 == 0: # p1's utterance
                        if token.lemma_ in self.p2_ideas.keys() and token.lemma_ not in self.p1_adopted_ideas: # Idea proposed by p2 but not previously adopted by p1
                            sentiment = self.analyzer.polarity_scores(" ".join(token.sent.text))["compound"]
                            if abs(sentiment - self.p2_ideas[token.lemma_]) < .1:
                                self.p1_adopted_ideas.add(token.lemma_)
                        elif token.lemma_ not in self.p2_ideas.keys():
                            sentiment = self.analyzer.polarity_scores(" ".join(token.sent.text))["compound"]
                            self.p1_ideas[token.lemma_] = sentiment
                    else: # p2's utterance
                        if token.lemma_ in self.p1_ideas.keys() and token.lemma_ not in self.p2_adopted_ideas: # Idea proposed by p1 but not previously adopted by p2
                            sentiment = self.analyzer.polarity_scores(" ".join(token.sent.text))["compound"]
                            if abs(sentiment - self.p1_ideas[token.lemma_]) < .1:
                                self.p2_adopted_ideas.add(token.lemma_)
                        elif token.lemma_ not in self.p1_ideas.keys():
                            sentiment = self.analyzer.polarity_scores(" ".join(token.sent.text))["compound"]
                            self.p2_ideas[token.lemma_] = sentiment
        return {
            "participant_1": {
                "num_ideas_adopted": len(self.p1_adopted_ideas)
            },
            "participant_2": {
                "num_ideas_adopted": len(self.p2_adopted_ideas)
            }
        }