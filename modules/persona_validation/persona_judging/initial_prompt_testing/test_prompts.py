PROMPT_TREE = """
### You are a dialogue classifier. Your only source of knowledge about dialogue types are the descriptions and classification tree encoded below:

1. Persuasion: the initial situation of a persuasion dialogue is a conflict of opinions. The participant’s goal in a persuasion dialogue is to persuade the other party. The goal of the dialogue is to resolve or clarify issues.
2. Negotiation: the initial situation of a negotiation dialogue is a conflict of interest. The participant’s goal in a negotiation dialogue is to get what they most want. The goal of the dialogue is to achieve a reasonable settlement both can live with.
3. Inquiry: the initial situation of an inquiry dialogue is a need to have proof. The participant’s goal in an inquiry dialogue is to find and verify evidence. The goal of the dialogue is to prove or disprove a hypothesis.
4. Information-seeking: the initial situation of an information-seeking dialogue is a need for information. The participant’s goal in an information-seeking dialogue is to acquire or give information. The goal of the dialogue is to exchange information.
5. Deliberation: the initial situation of a deliberation dialogue is a dilemma or practical choice. The participant’s goal in a deliberation dialogue is co-ordinate goals and actions. The goal of the dialogue is to decide the best available course of action.
6. Eristic: the initial situation of an eristic dialogue is a personal conflict. The participant’s goal in an eristic dialogue is to verbally hit out at the opponent. The goal of the dialogue is to reveal a deeper basis of conflict.

Is there a conflict?
    If there is a conflict, then is resolution the goal?
        If resolution is the goal, then classify the dialogue as “persuasion” 
        If resolution is not the goal, then is settlement the goal?
            If settlement is the goal, then classify the dialogue as “negotiation”
            If settlement is not the goal, then classify the dialogue as “eristic”
If there is not a conflict, then is there a common problem to be solved?
    If there is not a common problem to be solved, then classify the dialogue as “information-seeking”
    If there is a common problem to be solved, then is the problem theoretical?
        If the problem is theoretical, then classify the dialogue as “inquiry”
        If the problem is not theoretical, then classify the dialogue as “deliberation”


### Your task:
1. Read and parse the provided array of strings, where each string is a sequential utterance made by a participant in a conversation (only one party’s utterances are included).
2. Using exclusively the descriptions and classification tree above and no outside knowledge, classify which type of dialogue the participant is engaging in across all utterances.
3. Output ONLY one dialogue type which you have determined the participant to be engaging in (e.g., persuasion, negotiation, inquiry, information-seeking, deliberation, or eristic). Only choose one. 

### Participant utterances:
{{participant_utterances}}
"""

from google import genai
import json, time
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from google.genai import errors

CONVERSATIONS_PATH = "./modules/persona_validation/persona_judging/conversations_to_judge.json"
OUTPUT_PATH = "./modules/persona_validation/persona_judging/initial_prompt_testing/tree_desc_results.json"

@retry(
    stop=stop_after_attempt(8),
    wait=wait_random_exponential(multiplier=2, min=10, max=120),
    retry=retry_if_exception_type(errors.ServerError),
    reraise=True
)
def judge_conversation(conversation, prompt, client):
    messages_str = str(conversation["messages"])
    _prompt = prompt.replace("{{participant_utterances}}", messages_str)
    response = client.models.generate_content(model="gemini-3-flash-preview", contents=_prompt)
    return response.text.lower().replace("-", "_")

if __name__ == "__main__":
    with open(CONVERSATIONS_PATH) as f:
        data = json.load(f)

    heldout_set = data["test"]
    client = genai.Client()
    with open(OUTPUT_PATH) as f:
        heldout_judgements = json.load(f)
    
    for seed_id, convo_instances in heldout_set.items():
        if seed_id not in heldout_judgements:
            heldout_judgements[seed_id] = {}
        
        seed_judgments = heldout_judgements[seed_id]
        
        for convo_instance in convo_instances:
            persona = convo_instance["persona"]
            if persona == "discovery":
                continue
            if persona not in seed_judgments:
                seed_judgments[persona] = []
            
            existing_count = len(seed_judgments[persona])
            
            for iteration_idx in range(existing_count, 10):
                prediction = judge_conversation(convo_instance, PROMPT_TREE, client)
                seed_judgments[persona].append(prediction)
                print("=" * 80)
                print(f"Judging: seed {seed_id}, persona {persona}, iteration {iteration_idx + 1}")
                print("Prediction:", prediction)
                
                heldout_judgements[seed_id] = seed_judgments
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(heldout_judgements, f, indent=4)
            
    with open(OUTPUT_PATH, "w") as f:
        json.dump(heldout_judgements, f, indent=4)