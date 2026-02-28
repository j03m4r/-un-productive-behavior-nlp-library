import json
import random

def inspect_llm_conversations():
    LLM_CONVERSATIONS_PATH = "./llm_persona_conversations.json"

    results = None
    with open(LLM_CONVERSATIONS_PATH, "r") as f:
        results = json.load(f)

    persona_utterances = {
        "persuasion": [],
        "negotiation": [],
        "inquiry": [],
        "information-seeking": [],
        "discovery": [],
        "eristic": [],
        "deliberation": []
    }

    conversation_idxs = {
        "persuasion": None,
        "negotiation": None,
        "inquiry": None,
        "information-seeking": None,
        "discovery": None,
        "eristic": None,
        "deliberation": None
    }

    for idx, persona in enumerate(persona_utterances.keys()):
        rand_idx = random.randint(idx * 100, 99 + (idx * 100))
        conversation_idxs[persona] = rand_idx

    for key, utterances in persona_utterances.items():
        conversation = results[conversation_idxs[key]]
        for turn in conversation["messages"]:
            if turn["role"] == "assistant":
                utterances.append(turn["content"])
    
    for key, utterances in persona_utterances.items():
        print("\n" + "="*50)
        print(f"Persona: {key}")
        print("="*50)
        for idx, utterance in enumerate(utterances):
            print(f"Utterance {idx + 1}: {utterance}")

if __name__ == "__main__":
    inspect_llm_conversations()