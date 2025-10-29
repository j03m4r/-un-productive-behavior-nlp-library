from dotenv import load_dotenv
from modules import Evaluator
from textstat import smog_index
from pathlib import Path
import json, subprocess
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
                r_script = Path(__file__).resolve().parent / "r" / "politeness_eval.R"
                cmd = ["Rscript", str(r_script), utterance]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0 or not result.stdout.strip():
                    print("R STDERR:\n", result.stderr)
                    print("R STDOUT:\n", result.stdout)
                    raise RuntimeError(f"Rscript failed (exit {result.returncode})")
                politeness_dict = json.loads(result.stdout)
                res["utterances"].append(politeness_dict)

                for key in politeness_dict.keys(): # averaging for each respective feature, maybe we should just sum all features into one number
                    res['aggregate'][key] = res['aggregate'].get(key, 0) + politeness_dict[key] 

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

if __name__ == "__main__":
    conversation = [
        """Rich people pay too much in taxes. Period.

    I’m tired of hearing the same nonsense: “The rich don’t pay taxes because of loopholes.” That’s just not true. It’s lazy thinking and, frankly, intellectually dishonest.

    Let’s break this down.

    First off: when a company earns money, it pays taxes on those earnings, we’re talking 21% federal tax before a dollar goes to shareholders. Then, if that company pays a dividend, guess what? You (the shareholder) pay tax again on that same dollar you already indirectly paid tax on. That’s double taxation. And it doesn’t stop there.

    Let’s say you reinvest those earnings, grow your portfolio over time, and now you want to leave it to your kids. You’ve already paid taxes at the corporate level, then personal dividend/capital gains level… and now your kids get taxed again when they inherit it? That’s triple taxation. Insane.

    And capital gains? You’re only taxed when you sell. So sure, some people defer that. But they still pay. And when they do, it’s often on decades of compounding. That’s not a “loophole.” That’s smart investing and delayed gratification, the opposite of what most people do.

    I’m not saying the system’s perfect. But don’t buy into the lazy narrative that wealthy people are somehow dodging everything. Most high earners do pay an enormous amount, not just in income taxes, but on investment returns, estates, and more.

    If you want to argue about fairness, fine. But let’s at least get the facts straight first.""",
        """In capitalism, the metric for earning capital is labor. The more productive one's labor, the more wealth they are rewarded with. How productive are shareholders? Do they actively produce anything of value? I would argue that no, they do not. And yet, shareholders hold a vastly disproportionate amount of wealth. Naturally, we should tax them significantly and give this taxed wealth to the people who actually produce.""",
        """What naive garbage. Shareholders literally provide the CAPITAL that makes production possible in the first place. Without their investment, your precious "productive laborers" would be sitting around with zero tools, factories, or resources to be "productive" with. 
    Risk capital IS productive value - it's what transforms ideas into reality. Your fantasy redistribution scheme would destroy the very foundation that creates jobs and wealth for everyone.""",
        """I am not saying that shareholders should have no say in the functions of a capitalistic society (Whether or not I agree with capitalism is a different story). They just shouldn't be in possession of the vast majority of wealth when they aren't the ones doing the labor.""",
        """You're missing the point entirely. Shareholders don't just "possess" wealth - they EARNED it by risking their capital when others wouldn't. That's labor too, just intellectual and financial labor. 
    Meanwhile, workers get guaranteed paychecks regardless of company performance. Shareholders? They can lose everything overnight. Higher risk, higher reward. Your "fair share" nonsense ignores that shareholders already bear all the downside risk while workers face none.""",
        """They did not earn wealth. They pay laborers to create wealth, then take the vast majority of it for themselves. This is not even to mention the entrenched corporate interests that have bailed out shareholders at many points throughout American history (e.g. the housing market crash). So no, there is very little risk. Also corporations can fire workers without consequence if it appears better to shareholders. It is a far less risky business to be a shareholder."""
    ]

    ensemble_evaluator = EnsembleEvaluator()
    result = ensemble_evaluator.evaluate_conversation(conversation)
    print(result)
