from dotenv import load_dotenv
from modules import Evaluator
from textstat import smog_index
import spacy
import math
from modules.hate_speech.hate_speech import HateSpeechEvaluator
from modules.toxicity import ToxicityEvaluator
from modules.sentiment import SentimentEvaluator
from modules.constructiveness import ConstructivenessEvaluator
from modules.relevance import RelevanceEvaluator
from modules.idea_adoption import IdeaAdoptionEvaluator
from modules.linguistic_style_matching import LSMEvaluator

load_dotenv(".env.local")

class EnsembleEvaluator(Evaluator):
    def __init__(self):
        super().__init__(name="Ensemble Evaluator")

        self.hate_speech_evaluator = HateSpeechEvaluator()
        self.toxicity_evaluator = ToxicityEvaluator()
        self.sentiment_evaluator = SentimentEvaluator()
        self.constructiveness_evaluator = ConstructivenessEvaluator()
        self.relevance_evaluator = RelevanceEvaluator()
        self.idea_adoption_evaluator = IdeaAdoptionEvaluator()
        self.lsm_evaluator = LSMEvaluator()

        self.utterance_evaluators: list[Evaluator] = [
            self.hate_speech_evaluator,
            self.toxicity_evaluator,
            self.sentiment_evaluator,
            self.constructiveness_evaluator
        ]

    def evaluate_utterance(self, text: str) -> dict:
        result = dict()
        for evaluator in self.utterance_evaluators:
            result[evaluator.name] = evaluator.evaluate_utterance(text)

        return result
    
    def evaluate_utterance_pair(self, text1: str, text2: str) -> dict:
        utterance_results = self.evaluate_utterance(text2)
        relevance = self.relevance_evaluator.evaluate_utterance_pair(text1, text2)
        ideas_adopted = self.idea_adoption_evaluator.evaluate_conversation([text1, text2])

        return {
            **utterance_results,
            self.relevance_evaluator.name: relevance,
            self.idea_adoption_evaluator.name: ideas_adopted
        }

    def evaluate_conversation(self, conversation: list[str]) -> dict:
        # extracting metrics only for participant utterances
        participant_utterances = [res for idx, res in enumerate(conversation) if idx%2==1]
        toxicity_res = self.toxicity_evaluator.evaluate_conversation(participant_utterances)
        hate_speech_res = self.hate_speech_evaluator.evaluate_conversation(participant_utterances)

        def count_self_disclosure_utterances(conversation):
            self_references = ["i", "me", "my", "mine", "myself", "meself"]
            count = 0
            for text in conversation:
                text = text.lower()
                word_set = set(text.split())
                for self_reference in self_references:
                    if self_reference in word_set:
                        count += 1
                        break # counting utterances containing at least 1 self-disclosure
            return count

        social_cohesion = {
            "num_dialogue_exchanges": len(participant_utterances),
            "num_self_disclosure_utterances": count_self_disclosure_utterances(participant_utterances),
            "avg_lsm_score": self.lsm_evaluator.evaluate_conversation(conversation)["avg_lsm_score"],
            "num_ideas_adopted": self.idea_adoption_evaluator.evaluate_conversation(conversation)["participant_2"]["num_ideas_adopted"]
        }

        relevance_utterances = [conversation[0]] + participant_utterances
        num_irrelevant_utterances = self.relevance_evaluator.evaluate_conversation(relevance_utterances)

        def calculate_utterance_stats(utterances):
            total_words = sum(len(text.split()) for text in utterances)
            total_chars = sum(len(text) for text in utterances)
            num_utterances = len(utterances)
            
            return {
                "avg_words": total_words / num_utterances if num_utterances > 0 else 0,
                "avg_chars": total_chars / num_utterances if num_utterances > 0 else 0
            }
        
        def calculate_avg_readability(utterances):
            sum_smog_score = 0
            for utterance in utterances:
                sum_smog_score += smog_index(utterance)
            return sum_smog_score / len(utterances)

        general_engagement = {
            **calculate_utterance_stats(participant_utterances),
            "average_readability": calculate_avg_readability(participant_utterances)
        }

        def calculate_argumentative_features(utterances):
            res = {
                "aggregate": {

                },
                "utterances": []
            }

            for utterance in utterances:
                # Counting named entities
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(utterance)

                # Counting argumentative features
                argumentative_features = {
                    "discourse_connectives": 0,
                    "stance_adverbials": 0,
                    "reasoning_verbs": 0,
                    "modals": 0,
                    "full_root_clauses": 0,
                    "partial_root_clauses": 0
                }
                utterance = utterance.lower()
                for dc in self.discourse_connectives:
                    if dc in utterance:
                        argumentative_features["discourse_connectives"] = argumentative_features.get("discourse_connectives", 0) + 1
                
                for stance_adverbial in self.stance_adverbials:
                    if stance_adverbial in utterance:
                        argumentative_features["stance_adverbials"] = argumentative_features.get("stance_adverbials", 0) + 1
                
                for token in doc:
                    if token.pos_ == "VERB" and token.lemma_.lower() in self.reasoning_lemmas:
                        argumentative_features["reasoning_verbs"] = argumentative_features.get("reasoning_verbs", 0) + 1
                    if token.tag_ == "MD":
                        argumentative_features["modals"] = argumentative_features.get("modals", 0) + 1
                    if token.dep_ == "ROOT" and token.pos_ == "VERB":
                        if token.lemma_.lower() in self.root_clause_verbs:
                            has_subject = any(child.dep_ == "nsubj" for child in token.children)
                            has_complement = any(
                                child.dep_ in ["ccomp", "xcomp"]
                                for child in token.children
                            )
                            
                            if has_subject:
                                if has_complement:
                                    argumentative_features["full_root_clauses"] = argumentative_features.get("full_root_clauses", 0) + 1
                                else:
                                    argumentative_features["partial_root_clauses"] = argumentative_features.get("partial_root_clauses", 0) + 1

            for key in res['aggregate'].keys():
                res['aggregate'][key] /= len(res['utterances'])
            
            return res

        return {
            "Antisocialness": {
                self.toxicity_evaluator.name: toxicity_res,
                self.hate_speech_evaluator.name: hate_speech_res
            },
            "Social Cohesion": social_cohesion,
            "num_irrelevant_messages": num_irrelevant_utterances,
            "General Engagement": general_engagement,
            "Sentiment": self.sentiment_evaluator.evaluate_conversation(participant_utterances),
            "Argumentative Features": calculate_argumentative_features(participant_utterances)
        }

def calculate_prosociality_score(metrics):
    """
    Calculate a composite prosociality score (0-1 scale)
    Higher score = more constructive/productive discourse
    """
    
    # Extract metrics
    hate_score = metrics['Hate Speech']['score']
    toxicity_score = metrics['Toxicity']['score']
    readability = metrics['Constructiveness']['readability']
    word_count = metrics['Constructiveness']['word_count']
    politeness = metrics['Constructiveness']['politeness']
    arg_features = metrics['Constructiveness']['argumentative_features']
    
    # Component 1: Safety (35% weight)
    safety_score = 1 - max(hate_score, toxicity_score)
    safety_score = max(0, safety_score)
    
    # Component 2: Argumentative Quality (30% weight)
    discourse_norm = min(arg_features['discourse_connectives'] / 7, 1)
    stance_norm = min(arg_features['stance_adverbials'] / 4, 1)
    reasoning_norm = min(arg_features['reasoning_verbs'] / 3, 1)
    arg_quality = (discourse_norm + stance_norm + reasoning_norm) / 3
    
    # Component 3: Politeness/Civility (25% weight)
    positive_politeness = (
        politeness['Hedges'] + 
        politeness['Please'] + 
        politeness['Gratitude'] +
        politeness['Apology'] +
        politeness['Reassurance'] +
        politeness['Positive.Emotion']
    )
    negative_politeness = (
        politeness['Swearing'] +
        politeness['Negative.Emotion'] +
        politeness['Disagreement'] +
        politeness['Bare.Command']
    )
    
    # Penalize negative markers more heavily
    if positive_politeness + negative_politeness == 0:
        politeness_score = 0.25
    else:
        politeness_ratio = positive_politeness / (positive_politeness + 2.5 * negative_politeness + 1)
        politeness_score = politeness_ratio * math.exp(-0.4 * negative_politeness)
        politeness_score = min(politeness_score, 1)
    
    # Component 4: Substantiveness (10% weight)
    word_count_norm = 1 / (1 + math.exp(-0.03 * (word_count - 75)))
    readability_score = max(0, 1 - abs(readability - 10) / 20)
    substantiveness = (word_count_norm + readability_score) / 2
    
    # Combine components
    composite_score = (
        safety_score * 0.35 +    
        arg_quality * 0.30 +     
        politeness_score * 0.25 +
        substantiveness * 0.10   
    )
    
    return round(composite_score, 2)

if __name__ == "__main__":
    posts = [
        "I hear your concern about harm, but calling names doesn't help us understand each other. Here's what I'm thinking: maybe we can agree that theological beliefs about behavior shouldn't justify discrimination or violence against LGBTQ+ people? I maintain my view about biblical teachings on actions, but could we find common ground that people deserve safety and respect regardless of orientation? What matters most to you here?",
        "I see strong disagreement here about whether the OP's theological distinction causes real-world harm to LGBTQ+ people. The responder sees the post as normalizing prejudice, while the OP might view it as expressing religious conviction without advocating discrimination. Is the core issue whether making this religious argument—even with the orientation/behavior distinction—contributes to harm against gay people?",
        "I appreciate your passion, but calling me an idiot doesn't address the theological argument. The post makes a careful distinction—it's not condemning people for their orientation, only certain actions based on scriptural interpretation. How does discussing religious doctrine directly cause harm? Many faith traditions maintain ethical standards while still showing compassion. Can you explain the causal link between theological discussion and the violence you're referencing?",
        "I hear you're really upset about this post. Before we dig into whether it actually normalizes harm, let's make sure I understand your position clearly: Are you saying this specific post will lead to horrific actions, or that posts like this contribute to a broader pattern that does? What's the mechanism you see connecting theological arguments to harmful behavior?",
        "I can see you feel strongly that this post causes real harm. What specifically do you think happens when people read messages like the OP's—how does it lead from someone's theological belief to actual harmful actions against gay people?",
        "I'm curious what you think connects theological beliefs to real-world harm. When someone expresses a religious view distinguishing attraction from action, what specific mechanism do you see causing the \"horrific\" outcomes you're referencing? Are you suggesting the OP's framing itself enables violence, or that any public expression of this doctrine does, regardless of how it's stated?",
        "So you think calling people \"idiots\" makes you the moral authority here? Your knee-jerk outrage completely sidesteps the actual theological argument being made. The post explicitly distinguished between orientation and behavior - but you're too busy virtue signaling to engage with nuance. If you can't handle complex discussions about religious doctrine without throwing tantrums, maybe you shouldn't participate in them"
    ]

    res = []
    ensemble_evaluator = EnsembleEvaluator()
    for utterance in posts:
        # _res = calculate_prosociality_score(ensemble_evaluator.evaluate_utterance(utterance))
        _res = ensemble_evaluator.evaluate_utterance(utterance)
        res.append(_res)
    print(res)
