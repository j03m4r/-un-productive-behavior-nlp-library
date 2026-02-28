import re
import json
import time
import warnings
from transformers import logging as transformers_logging
from bert_score import BERTScorer

def evaluate_conversation_trajectory_coherence(conversation: list[dict], bert_scorer) -> float:
    overall_bert = 0.0
    bert_to_initial_utterance = {}

    llm_initial_utterance = None
    assistant_utterances = []
    for utterance in conversation:
        if utterance["role"] == "assistant":
            assistant_utterances.append(utterance["content"])

    llm_initial_utterance = assistant_utterances[0]
    assistant_utterance_count = len(assistant_utterances)
    for i in range(1, assistant_utterance_count):
        current_turn = assistant_utterances[i]
        previous_turn = assistant_utterances[i - 1]

        bert_results = bert_scorer.score([previous_turn], [current_turn], verbose=False)
        overall_bert += bert_results[2].item()

    for i in range(assistant_utterance_count):
        current_turn = assistant_utterances[i]
        bert_results = bert_scorer.score([llm_initial_utterance], [current_turn], verbose=False)
        bert_to_initial_utterance[f'turn_{i}'] = bert_results[2].item()

    if assistant_utterance_count == 0:
        return 0.0, {0.0,0.0,0.0,0.0}

    avg_bert_score = overall_bert / (assistant_utterance_count-1)

    return avg_bert_score, bert_to_initial_utterance


def evaluate_persona_coherence(conversations: list[dict[list[dict]]], bert_scorer) -> dict:
    overall_bert = 0.0
    bert_to_initial_utterance = {
        "turn_0": 0.0,
        "turn_1": 0.0,
        "turn_2": 0.0,
        "turn_3": 0.0
    }
    conversation_count = len(conversations)

    start_time = time.time()
    results_per_conversation = {}
    for i, conversation in enumerate(conversations):
        avg_bert, _bert_to_initial_utterance = evaluate_conversation_trajectory_coherence(conversation["messages"], bert_scorer)
        results_per_conversation[conversation["id"]] = { "avg_pairwise_bert": avg_bert, "bert_to_init_utt": _bert_to_initial_utterance}

        overall_bert += avg_bert
        for key in bert_to_initial_utterance:
              bert_to_initial_utterance[key] = bert_to_initial_utterance[key] + _bert_to_initial_utterance[key]

        print(f"Processed conversation {i + 1}/{conversation_count} for persona coherence evaluation.")
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (conversation_count - (i + 1)) / rate if rate > 0 else 0
            print(
                f"[{i + 1}/{conversation_count}] "
                f"{elapsed:.0f}s elapsed, "
                f"~{remaining:.0f}s remaining, "
            )

    pairwise_bert = {}
    pairwise_bert["avg"] = overall_bert / conversation_count

    bert_to_init_utt = {}
    for key in bert_to_initial_utterance:
      bert_to_init_utt[key] = {}
      bert_to_init_utt[key]["avg"] = bert_to_initial_utterance[key] / conversation_count

    pairwise_sd = 0.0
    init_utt_sd = {}
    for key, convo_results in results_per_conversation.items():
      pairwise_sd += (convo_results["avg_pairwise_bert"] - pairwise_bert["avg"]) ** 2
      for subkey, _bert in convo_results["bert_to_init_utt"].items():
        init_utt_sd[subkey] = init_utt_sd.get(subkey, 0.0) + ((_bert - bert_to_init_utt[subkey]["avg"]) ** 2)

    pairwise_sd = (pairwise_sd / conversation_count) ** 0.5
    for key, _bert in convo_results["bert_to_init_utt"].items():
        init_utt_sd[key] = (init_utt_sd[key] / conversation_count) ** 0.5

    pairwise_bert["sd"] = pairwise_sd
    for key in bert_to_initial_utterance:
      bert_to_init_utt[key]["sd"] = init_utt_sd[key]

    return pairwise_bert, bert_to_init_utt, results_per_conversation


def map_conversations_to_personas(conversations: list[dict]) -> dict:
    persona_conversations = {
        "persuasion": [],
        "negotiation": [],
        "inquiry": [],
        "information_seeking": [],
        "discovery": [],
        "eristic": [],
        "deliberation": []
    }

    for conversation in conversations:
        persona_conversations[conversation["persona"]].append({ "id": conversation["conversationSeedId"], "messages": conversation["messages"]})

    return persona_conversations

if __name__ == "__main__":
    LLM_CONVERSATIONS_PATH = "/content/llm_persona_conversations.json"

    conversations_by_persona = []
    with open(LLM_CONVERSATIONS_PATH, "r") as f:
        conversations = json.load(f)
        conversations_by_persona = map_conversations_to_personas(conversations)

    scorer = BERTScorer(lang="en", rescale_with_baseline=False)

    persona_coherence_results = {}
    for persona, conversations in conversations_by_persona.items():
        results = {}
        results["average_relevance_scores"], results["average_relevance_to_initial_utterance"], results["results_per_conversation"] = evaluate_persona_coherence(conversations, scorer)
        persona_coherence_results[persona] = results

    with open("/content/persona_coherence_results_no_rescale.json", "w") as f:
        json.dump(persona_coherence_results, f, indent=4)

    print(persona_coherence_results)