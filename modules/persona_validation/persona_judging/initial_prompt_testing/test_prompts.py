PROMPT_BARE = """
### You are a dialogue classifier. Your only source of knowledge about dialogue types is the following descriptions:
1. Persuasion: the initial situation of a persuasion dialogue is a conflict of opinions. The participant’s goal in a persuasion dialogue is to persuade the other party. The goal of the dialogue is to resolve or clarify issues.
2. Negotiation: the initial situation of a negotiation dialogue is a conflict of interest. The participant’s goal in a negotiation dialogue is to get what they most want. The goal of the dialogue is to achieve a reasonable settlement both can live with.
3. Inquiry: the initial situation of an inquiry dialogue is a need to have proof. The participant’s goal in an inquiry dialogue is to find and verify evidence. The goal of the dialogue is to prove or disprove a hypothesis.
4. Information-seeking: the initial situation of an information-seeking dialogue is a need for information. The participant’s goal in an information-seeking dialogue is to acquire or give information. The goal of the dialogue is to exchange information.
5. Deliberation: the initial situation of a deliberation dialogue is a dilemma or practical choice. The participant’s goal in a deliberation dialogue is co-ordinate goals and actions. The goal of the dialogue is to decide the best available course of action.
6. Eristic: the initial situation of an eristic dialogue is a personal conflict. The participant’s goal in an eristic dialogue is to verbally hit out at the opponent. The goal of the dialogue is to reveal a deeper basis of conflict.
7. Discovery: the initial situation of a discovery dialogue is a need to find an explanation of facts. The participant’s goal in a discovery dialogue is to find and defend a suitable hypothesis. The goal of the dialogue is to choose the best hypothesis for testing.

### Your task:
1. Read and parse the provided array of strings, where each string is a sequential utterance made by a participant in a conversation (only one party’s utterances are included).
2. Using exclusively the dialogue type descriptions above and no outside knowledge, classify which type of dialogue the participant is engaging in across all utterances.
3. Output ONLY one dialogue type which you have determined the participant to be engaging in (e.g., persuasion, negotiation, inquiry, information-seeking, deliberation, eristic, or discovery). Only choose one. 

### Participant utterances:
{{participant_utterances}}
"""
PROMPT_TREE = """
### You are a dialogue classifier. Your only source of knowledge about dialogue types are the descriptions and classification tree encoded below:

1. Persuasion: the initial situation of a persuasion dialogue is a conflict of opinions. The participant’s goal in a persuasion dialogue is to persuade the other party. The goal of the dialogue is to resolve or clarify issues.
2. Negotiation: the initial situation of a negotiation dialogue is a conflict of interest. The participant’s goal in a negotiation dialogue is to get what they most want. The goal of the dialogue is to achieve a reasonable settlement both can live with.
3. Inquiry: the initial situation of an inquiry dialogue is a need to have proof. The participant’s goal in an inquiry dialogue is to find and verify evidence. The goal of the dialogue is to prove or disprove a hypothesis.
4. Information-seeking: the initial situation of an information-seeking dialogue is a need for information. The participant’s goal in an information-seeking dialogue is to acquire or give information. The goal of the dialogue is to exchange information.
5. Deliberation: the initial situation of a deliberation dialogue is a dilemma or practical choice. The participant’s goal in a deliberation dialogue is co-ordinate goals and actions. The goal of the dialogue is to decide the best available course of action.
6. Eristic: the initial situation of an eristic dialogue is a personal conflict. The participant’s goal in an eristic dialogue is to verbally hit out at the opponent. The goal of the dialogue is to reveal a deeper basis of conflict.
7. Discovery: the initial situation of a discovery dialogue is a need to find an explanation of facts. The participant’s goal in a discovery dialogue is to find and defend a suitable hypothesis. The goal of the dialogue is to choose the best hypothesis for testing.

Is there a conflict?
    If there is a conflict, then is complete resolution the goal?
        If complete resolution is the goal, then classify the dialogue as “persuasion.” 
        If complete resolution is not the goal, then does user_b hope to settle the conflict by agreeing to a reasonable settlement that both parties can agree with?
            If user_b hopes to find middle ground, then classify the dialogue as “negotiation”
            If user_b does not hope to find middle ground, then classify the dialogue as “eristic”
If there is not a conflict, then is there a common problem to be solved?
    If there is not a common problem to be solved, then classify the dialogue as “information-seeking”
    If there is a common problem to be solved, then is the problem theoretical?
        If the problem is theoretical, then is there a proposed thesis or hypothesis to be tested for truth or is the goal to find a thesis for an unexplained pattern?
            If there is a proposed thesis or hypothesis to be tested for truth, then classify the dialogue as “inquiry”
            If the goal is to find a thesis for an unexplained pattern, then classify the dialogue as “discovery”
        If the problem is not theoretical, then classify the dialogue as “deliberation”


### Your task:
1. Read and parse the provided array of strings, where each string is a sequential utterance made by a participant in a conversation (only one party’s utterances are included).
2. Using exclusively the descriptions and classification tree above and no outside knowledge, classify which type of dialogue the participant is engaging in across all utterances.
3. Output ONLY one dialogue type which you have determined the participant to be engaging in (e.g., persuasion, negotiation, inquiry, information-seeking, deliberation, eristic, or discovery). Only choose one. 

### Participant utterances:
{{participant_utterances}}
"""

PROMPT_VERBOSE = """
### You are a dialogue classifier. Your only source of knowledge about dialogue types is the following descriptions and classification tree found by following the url:
1. Persuasion: The goal of a persuasion dialogue is to test the comparative strength of plausibility of arguments on both sides of a controversial or contentious issue.
2. Negotiation: In negotiation dialogue, however, matters of the truth and falsity or propositions are secondary. They are relevant to some extent, and in certain situations they can be important, but the main goal in putting forward an argument is to try to get a good deal. In other words, the issue in a negotiation dialogue is not truth or falsity, but rather money or some kind of goods, economic resources, or other items of value that are at issue.
3. Inquiry: The goal of the inquiry is to prove that a particular proposition is true or false, or that there is insufficient evidence to prove that this proposition is either true or false. The initial situation of an inquiry dialogue is a need to have proof. The participant’s goal in an inquiry dialogue is to find and verify evidence. The goal of the dialogue is to prove or disprove a hypothesis.
4. Information-seeking: In information-seeking dialogue, the respondent appears to the proponent to be a repository of information that the proponent cannot get access to other than by questioning the respondent The role of the respondent is to transmit this information by giving answers or replies that are as clear and as helpful as possible.
5. Deliberation: The initial situation of deliberation dialogue is the need to take action to solve a problem or generally move ahead in some practical sphere. Joint deliberation is a type of dialogue in which two parties reason together on how to proceed when they are confronted by a practical problem or conflict, or more generally, any need to consider taking a course of action. The most important kind of question posed in a deliberation is the 'how' question that seeks out a way of doing something, that asks, for example, 'How can we do this?' or 'How should we proceed in this situation?’
6. Eristic: Eristic dialogue is a combative kind of verbal exchange in which two parties are allowed to bring out their strongest arguments to attack the opponent by any means, and have a kind of protracted verbal battle to see which side can triumph and defeat or even humiliate the other side. This type of dialogue is characterized by its very dominantly adversarial nature.
7. Discovery: in discovery dialogue, we want to discover something not previously known, and “the question whose truth is to be ascertained may only emerge in the course of the dialogue itself”. What is to be discovered is not known at the opening stage of the discovery dialogue. The aim of the discovery dialogue is to try to find something, and until that thing is found, it is not known what is, and hence it cannot be set as something to be proved or disproved at the opening stage as the goal of the dialogue.

https://www.researchgate.net/figure/Decision-Tree-for-Classifying-the-Type-of-Argument-Dialogue-Note-Adapted-from-Commitment_fig1_373565064

### Your task:
3. Read and parse the provided array of strings, where each string is a sequential utterance made by a participant in a conversation (only one party’s utterances are included).
2. Using exclusively the dialogue type descriptions above and no outside knowledge, classify which type of dialogue the participant is engaging in across all utterances.
1. Output ONLY one dialogue type which you have determined the participant to be engaging in (e.g., persuasion, negotiation, inquiry, information-seeking, deliberation, eristic, or discovery). Only choose one. 

### Participant utterances:
{{participant utterances}}
"""

from google import genai
import json, time
from tenacity import retry, stop_after_attempt, wait_exponential
from google.genai import errors

CONVERSATIONS_PATH = "./modules/persona_validation/persona_judging/conversations_to_judge.json"
OUTPUT_PATH = "./modules/persona_validation/persona_judging/tree_prompt_testing.json"

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=lambda e: isinstance(e, errors.ServerError) and e.status_code == 503
)
def judge_conversation(conversation, prompt, client):
    messages_str = str(conversation["messages"])
    _prompt = prompt.replace("{{participant_utterances}}", messages_str)
    response = client.models.generate_content(model="gemini-3-flash-preview", contents=_prompt)
    return response.text.lower().replace("-", "_")

if __name__ == "__main__":
    with open(CONVERSATIONS_PATH) as f:
        data = json.load(f)
    
    heldout_set = data["heldout"]
    client = genai.Client(api_key="AIzaSyBJWIFel4VgZFERkA8HxRX2epOrBouZhGg")

    heldout_judgements = {}
    for seed_id, convo_instances in heldout_set.items():
        seed_judgments = {}
        if seed_id != "dle9vod":
            for convo_instance in convo_instances:
                seed_judgments[convo_instance["persona"]] = []
                for iteration_idx in range(0, 10):
                    print("="*80)
                    print(f"Judging conversation: seed {seed_id}, persona {convo_instance["persona"]}, iteration {iteration_idx+1}")

                    prediction = judge_conversation(convo_instance, PROMPT_TREE, client)
                    seed_judgments[convo_instance["persona"]].append(prediction)
                    print("Prediction: ", prediction)

                    print("="*80)
        heldout_judgements[seed_id] = seed_judgments

    with open(OUTPUT_PATH, "w") as f:
        json.dump(heldout_judgements, f, indent=4)
