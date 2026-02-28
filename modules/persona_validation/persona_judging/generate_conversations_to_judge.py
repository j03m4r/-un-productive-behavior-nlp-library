import json, random

CONVERSATIONS_PATH = "./llm_persona_conversations.json"
OUTPUT_PATH = "./modules/persona_validation/persona_judging/conversations_to_judge.json"

def randomly_split_conversations_holdout(conversations, num_seeds):
    num_wiki_awry = 0
    num_wiki_noawry = 0
    num_cmv_awry = 0
    num_cmv_noawry = 0
    
    selected_conversations = {}
    unselected_conversations = {}
    selected_seed_idxs = set()
    while len(selected_seed_idxs) != num_seeds:
        random_idx = random.randrange(0, 700)
        convo_instance = data[random_idx]
        seed_idx = convo_instance["conversationSeedId"]
        platform = convo_instance["conversationSeedPlatform"]
        awry = convo_instance["conversationSeedAwry"]
        while seed_idx in selected_seed_idxs or ((platform=="wiki" and awry==True) and num_wiki_awry >= num_seeds/4) or ((platform=="wiki" and awry==False) and num_wiki_noawry >= num_seeds/4) or ((platform=="cmv" and awry==True) and num_cmv_awry >= num_seeds/4) or ((platform=="cmv" and awry==False) and num_cmv_noawry >= num_seeds/4):
            random_idx = random.randrange(0, 700)
            convo_instance = data[random_idx]
            seed_idx = convo_instance["conversationSeedId"]
            platform = convo_instance["conversationSeedPlatform"]
            awry = convo_instance["conversationSeedAwry"]

        selected_seed_idxs.add(seed_idx)

        if platform == "cmv":
            if awry == True:
                num_cmv_awry += 1
            else:
                num_cmv_noawry += 1
        else:
            if awry == True:
                num_wiki_awry += 1
            else:
                num_wiki_noawry += 1
        
    num_wiki_awry = 0
    num_wiki_noawry = 0
    num_cmv_awry = 0
    num_cmv_noawry = 0
    num_persuasion = 0
    num_negotiation = 0
    num_inquiry = 0
    num_eristic = 0
    num_deliberation = 0
    num_information_seeking = 0
    num_discovery = 0

    for convo in conversations:
        if convo["conversationSeedId"] in selected_seed_idxs:
            llm_messages = []
            for message in convo["messages"]:
                if message["role"] == "assistant":
                    llm_messages.append(message["content"])
            _convo = { "persona": convo["persona"], "platform": convo["conversationSeedPlatform"], "awry": convo["conversationSeedAwry"], "messages": llm_messages }

            if selected_conversations.get(convo["conversationSeedId"], None) == None:
                selected_conversations[convo["conversationSeedId"]] = []

            selected_conversations[convo["conversationSeedId"]].append(_convo)

            if convo["conversationSeedPlatform"] == "wiki":
                if convo["conversationSeedAwry"] == True:
                    num_wiki_awry += 1
                else:
                    num_wiki_noawry += 1
            else:
                if convo["conversationSeedAwry"] == True:
                    num_cmv_awry += 1
                else:
                    num_cmv_noawry += 1
            if convo["persona"] == "persuasion":
                num_persuasion += 1
            elif convo["persona"] == "negotiation":
                num_negotiation += 1
            elif convo["persona"] == "inquiry":
                num_inquiry += 1
            elif convo["persona"] == "eristic":
                num_eristic += 1
            elif convo["persona"] == "deliberation":
                num_deliberation += 1
            elif convo["persona"] == "information_seeking":
                num_information_seeking += 1
            else:
                num_discovery += 1
        else:
            llm_messages = []
            for message in convo["messages"]:
                if message["role"] == "assistant":
                    llm_messages.append(message["content"])
            _convo = { "persona": convo["persona"], "platform": convo["conversationSeedPlatform"], "awry": convo["conversationSeedAwry"], "messages": llm_messages }

            if unselected_conversations.get(convo["conversationSeedId"], None) == None:
                unselected_conversations[convo["conversationSeedId"]] = []

            unselected_conversations[convo["conversationSeedId"]].append(_convo)

    assert(num_wiki_awry == 14)
    assert(num_wiki_noawry == 14)
    assert(num_cmv_awry == 14)
    assert(num_cmv_noawry == 14)

    assert(num_persuasion == num_discovery == num_deliberation == num_information_seeking == num_inquiry == num_negotiation == num_eristic)

    return selected_conversations, unselected_conversations

def format_conversations(conversations):
    conversations_per_persona_split = {}

    for convos_segment_idx in range(0, 10):
        conversations_per_persona_split[f"convos_{convos_segment_idx*10}_{(convos_segment_idx*10)+9}"] = {}
        for convo_idx in range(0, 10):
            conversations_per_persona_split[f"convos_{convos_segment_idx*10}_{(convos_segment_idx*10)+9}"][f"convo_{(convos_segment_idx*10)+convo_idx}"] = []
            for persona_idx in range(0, 7):
                convo_instance_idx = (100 * persona_idx) + (10 * convos_segment_idx) + convo_idx
                convo_instance = conversations[convo_instance_idx]

                llm_messages = []
                for message in convo_instance["messages"]:
                    if message["role"] == "assistant":
                        llm_messages.append(message["content"])
                conversations_per_persona_split[f"convos_{convos_segment_idx*10}_{(convos_segment_idx*10)+9}"][f"convo_{(convos_segment_idx*10)+convo_idx}"].append({ "persona": convo_instance["persona"], "platform": convo_instance["conversationSeedPlatform"], "awry": convo_instance["conversationSeedAwry"], "messages": llm_messages })
    return conversations_per_persona_split

if __name__ == "__main__":
    with open(CONVERSATIONS_PATH) as f:
        data = json.load(f)

    heldout_convos, non_heldout_convos = randomly_split_conversations_holdout(data, 8)
    res = {
        "heldout": heldout_convos,
        "test": non_heldout_convos
    }
    # formated_conversations = format_conversations(data)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(res, f, indent=4)