from dotenv import load_dotenv
from modules.constructiveness import ConstructivenessEvaluator
from modules.sentiment import SentimentEvaluator
from modules.toxicity import ToxicityEvaluator
import re
import json
import time
from pathlib import Path
from collections import defaultdict

first_person_pronouns = {"i", "me", "my", "mine", "myself"}
second_person_pronouns = {"you", "your", "yours", "yourself", "yourselves"}
third_person_pronouns = {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs", "himself", "herself", "itself", "themselves"}
inclusive_pronouns = {"we", "us", "our", "ours", "ourselves"}
negations = {
            "not", "nor", "no", "nowhere", "cannot", "never", "neither", "nah", "nope",
            "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", 
            "hadn't", "won't", "wouldn't", "don't", "doesn't", "didn't", 
            "can't", "couldn't", "shouldn't", "mightn't", "mustn't", "shan't"
        }

concessions = {"i agree", "i concede", "you're right", "you are right", "that's true", "that is true", "that's fair", "that is fair", "fair point", "fair enough", "good point", "valid point", "i'll grant", "i will grant", "i accept", "i acknowledge", "you make a good point", "you make a fair point", "you raise a good point", "i see your point", "i take your point", "point taken", "i stand corrected", "you have a point", "i hadn't considered", "i hadn't thought of", "that's a good point", "admittedly", "granted", "touché", "i can see that", "i can see how", "you're not wrong", "you are not wrong", "that's so true", "tell me about it", "absolutely", "i concur", "i partially agree", "i'm in partial agreement"}
challenges = {"i disagree", "that's wrong", "that is wrong", "you're wrong", "you are wrong", "that's not true", "that is not true", "that's false", "that doesn't make sense", "that doesn't follow", "how can you", "how do you", "that's ridiculous", "that's absurd", "that's nonsense", "you're missing", "you are missing", "you're ignoring", "you are ignoring", "you fail to", "that's a stretch", "that doesn't hold", "where's your evidence", "prove it", "says who", "based on what", "that contradicts", "you can't just", "that's not how", "on what basis", "what evidence", "you haven't shown", "you haven't demonstrated", "that logic doesn't", "by that logic", "i object", "i take issue", "i have an issue", "i have a problem", "i don't accept that", "i don't think so", "not necessarily", "that's not quite right", "don't be so sure", "yeah...no", "you must be joking", "you can't be serious", "i beg to differ", "i respectfully disagree", "that couldn't be further from the truth", "no way", "the exact opposite", "that's not always true", "that's not always the case", "i'm not so sure about that", "i can't see that", "i'm not convinced", "i don't entirely agree", "i cannot agree", "i'm afraid i can't agree on that assessment", "this is in complete contradiction"}
proposals = {"what if", "how about", "let's", "let us", "i suggest", "i propose", "perhaps we", "maybe we", "we could", "we should", "we might", "why don't we", "why not", "one option", "one approach", "an alternative", "here's what i think", "here is what i think", "i'd suggest", "i would suggest", "consider", "would you agree that", "can we agree", "could we agree", "suppose we", "imagine if", "how would you feel about", "what do you think about", "i think we should", "my suggestion", "my proposal", "let's consider", "let us consider", "i recommend", "i advise", "i encourage", "i'm in favor", "i'm all for", "i'm on board with"}
acknowledgments = {"i see", "i understand", "i hear you", "i hear what", "that makes sense", "interesting", "that's interesting", "that is interesting", "i appreciate", "thanks for", "thank you for", "i get what", "i get that", "i get your", "i see what you", "i see where you", "i follow", "i can see why", "i can understand", "noted", "understood", "i take note", "that's helpful", "good to know", "i see your perspective", "i understand your", "i appreciate your", "with all due respect", "i realize", "i respect", "thank you", "i'm grateful"}

def tokenize_with_contractions(text):
    """
    Handle cases like "I'm" -> ["i", "am"], "you're" -> ["you", "are"]
    """
    contractions = {
        "i'm": "i am",
        "you're": "you are", 
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "i'll": "i will",
        "you'll": "you will",
        "we'll": "we will",
        "i'd": "i would",
        "you'd": "you would",
        "we'd": "we would",
    }
    
    text_lower = text.lower()
    for contraction, expansion in contractions.items():
        text_lower = text_lower.replace(contraction, expansion)
    
    return re.findall(r'\b\w+\b', text_lower)

def clean_for_word_length(text):
    return re.findall(r"\b[\w']+\b", text.lower())

def extract_turn_features(utterance: str, constructiveness_evaluator: ConstructivenessEvaluator, sentiment_evaluator: SentimentEvaluator, toxicity_evaluator: ToxicityEvaluator) -> dict:
    words = tokenize_with_contractions(utterance)
    constructiveness_results = constructiveness_evaluator.evaluate_utterance(utterance)
    word_count = constructiveness_results["word_count"]
    sentence_count = utterance.count(".") + utterance.count("!") + utterance.count("?")
    avg_word_len = sum(len(word) for word in clean_for_word_length(utterance)) / (len(clean_for_word_length(utterance)) + 1e-5)
    question_count = utterance.count("?")
    exclamation_count = utterance.count("!")
    question_ratio = question_count / (sentence_count + 1e-5)
    first_person_ratio = sum(1 for word in words if word in first_person_pronouns) / (len(words) + 1e-5)
    second_person_ratio = sum(1 for word in words if word in second_person_pronouns) / (len(words) + 1e-5)
    inclusive_pronoun_ratio = sum(1 for word in words if word in inclusive_pronouns) / (len(words) + 1e-5)
    politeness = constructiveness_results["politeness"]
    discourse_connectives_count = constructiveness_results["argumentative_features"]["discourse_connectives"]
    stance_adverbials_count = constructiveness_results["argumentative_features"]["stance_adverbials"]
    reasoning_verbs_count = constructiveness_results["argumentative_features"]["reasoning_verbs"]
    modal_verbs_count = constructiveness_results["argumentative_features"]["modals"]
    full_root_clauses_count = constructiveness_results["argumentative_features"]["full_root_clauses"]
    partial_root_clauses_count = constructiveness_results["argumentative_features"]["partial_root_clauses"]
    negation_count = sum(1 for word in words if word in negations)
    concession_count = sum(1 for phrase in concessions if phrase in utterance.lower())
    challenge_count = sum(1 for phrase in challenges if phrase in utterance.lower())
    proposal_count = sum(1 for phrase in proposals if phrase in utterance.lower())
    acknowledgment_count = sum(1 for phrase in acknowledgments if phrase in utterance.lower())
    sentiment_results = sentiment_evaluator.evaluate_utterance(utterance)
    toxicity_results = toxicity_evaluator.evaluate_utterance(utterance)

    ret = {
        # Structural
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_len": avg_word_len,
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "question_ratio": question_ratio,
        
        # Pronoun ratios
        "first_person_ratio": first_person_ratio,
        "second_person_ratio": second_person_ratio,
        "inclusive_pronoun_ratio": inclusive_pronoun_ratio,
        
        # Argumentative structure
        "discourse_connectives_count": discourse_connectives_count,
        "stance_adverbials_count": stance_adverbials_count,
        "reasoning_verbs_count": reasoning_verbs_count,
        "modal_verbs_count": modal_verbs_count,
        "full_root_clauses_count": full_root_clauses_count,
        "partial_root_clauses_count": partial_root_clauses_count,
        "negation_count": negation_count,
        
        # Dialogue acts
        "concession_count": concession_count,
        "challenge_count": challenge_count,
        "proposal_count": proposal_count,
        "acknowledgment_count": acknowledgment_count,
        
        # Politeness (individual features)
        **{f"politeness_{k}": v for k, v in politeness.items()},

        # Sentiment
        **{f"sentiment_{k}": v for k, v in sentiment_results.items()},

        # Toxicity
        "toxicity_score": toxicity_results["score"],
    }

    return ret

def extract_conversation_features(conversation: dict, constructiveness_evaluator: ConstructivenessEvaluator, sentiment_evaluator: SentimentEvaluator, toxicity_evaluator: ToxicityEvaluator) -> dict:
    assistant_turns = [m for m in conversation["messages"] if m["role"] == "assistant"]
    turns = []

    for i, turn in enumerate(assistant_turns):
        turn_features = extract_turn_features(turn["content"], constructiveness_evaluator, sentiment_evaluator, toxicity_evaluator)
        turns.append({
            "turn_idx": i,
            "features": turn_features
        })
    return {
        "persona": conversation["persona"],
        "seed_id": conversation["conversationSeedId"],
        "platform": conversation["conversationSeedPlatform"],
        "has_attack": conversation["conversationSeedAwry"],
        "turns": turns
    }

CONVERSATIONS_PATH = "/home/joemar/Documents/research/argumentative_convo_mod/persona_testing/conversation_testbed/llm_persona_conversations.json"
OUTPUT_DIR = Path("./modules/persona_validation/output")
OUTPUT_PATH = OUTPUT_DIR / "extracted_features.json"

load_dotenv(".env.local")

def load_conversations(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversations")
    return data


def group_by_persona(features: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for f in features:
        grouped[f["persona"]].append(f)
    return dict(grouped)

def load_failed_conversation(path: str) -> dict:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if data:
            print(f"Loaded {len(data)} failed conversations from previous run")
            return data  # Just return the first one for reference
    except FileNotFoundError:
        print("No previous extraction failures found.")
    return None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load conversations
    # conversations = load_conversations(CONVERSATIONS_PATH)

    # 2. Initialize evaluator ONCE
    constructiveness_evaluator = ConstructivenessEvaluator()
    sentiment_evaluator = SentimentEvaluator()
    toxicity_evaluator = ToxicityEvaluator()

    # 3. Extract features for all conversations
    all_features = []
    #total = len(conversations)
    failed = []
    start_time = time.time()

    failed_conversation = load_failed_conversation("./modules/persona_validation/output/extraction_failures.json")
    features = extract_conversation_features(failed_conversation, constructiveness_evaluator, sentiment_evaluator, toxicity_evaluator)
    print(features)

    # for i, conversation in enumerate(conversations):
    #     try:
    #         features = extract_conversation_features(conversation, constructiveness_evaluator, sentiment_evaluator, toxicity_evaluator)
    #         all_features.append(features)
    #     except Exception as e:
    #         failed.append({
    #             "index": i,
    #             "persona": conversation.get("persona", "unknown"),
    #             "seed_id": conversation.get("conversationSeedId", "unknown"),
    #             "error": str(e)
    #         })

    #     # Progress logging every 50 conversations
    #     if (i + 1) % 50 == 0 or (i + 1) == total:
    #         elapsed = time.time() - start_time
    #         rate = (i + 1) / elapsed
    #         remaining = (total - (i + 1)) / rate if rate > 0 else 0
    #         print(
    #             f"[{i + 1}/{total}] "
    #             f"{elapsed:.0f}s elapsed, "
    #             f"~{remaining:.0f}s remaining, "
    #             f"{len(failed)} failures"
    #         )

    # # 4. Report failures
    # if failed:
    #     print(f"\n{len(failed)} conversations failed extraction:")
    #     for f in failed[:10]:
    #         print(f"  {f['persona']} / {f['seed_id']}: {f['error']}")
    #     if len(failed) > 10:
    #         print(f"  ... and {len(failed) - 10} more")

    #     # Save failures for debugging
    #     with open(OUTPUT_DIR / "extraction_failures.json", "w") as f:
    #         json.dump(failed, f, indent=2)

    # # 5. Group by persona and print summary
    # grouped = group_by_persona(all_features)
    # print(f"\nExtracted features for {len(all_features)} conversations:")
    # for persona, convos in sorted(grouped.items()):
    #     platforms = defaultdict(int)
    #     attack_counts = defaultdict(int)
    #     for c in convos:
    #         platforms[c["platform"]] += 1
    #         attack_counts[c["has_attack"]] += 1
    #     print(
    #         f"  {persona:25s}: {len(convos):3d} convos "
    #         f"(wiki={platforms['wiki']}, cmv={platforms['cmv']}, "
    #         f"attack={attack_counts[True]}, no_attack={attack_counts[False]})"
    #     )

    # # 6. Save to disk
    # output = {
    #     "metadata": {
    #         "total_conversations": len(all_features),
    #         "total_failed": len(failed),
    #         "personas": list(grouped.keys()),
    #         "extraction_time_seconds": round(time.time() - start_time, 1),
    #     },
    #     "by_persona": grouped,
    #     "all_features": all_features,
    # }

    # with open(OUTPUT_PATH, "w") as f:
    #     json.dump(output, f, indent=2)

    # print(f"\nSaved to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()