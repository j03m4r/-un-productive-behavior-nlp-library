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
        # Topic 1: Men should have a say in abortion
        "An abortion should require the consent of the father\nProvided there is no threat to the mother's life or health and the baby was conceived through consensual sex, if a woman opts to terminate the pregnancy, she should REQUIRE the consent of the father. The reality is the father's child too. I don't get why this is even controversial - the father should be prepared to raise the child alone if the mother won't. But it is still the father's child!! Men's rights are constantly ignored in this debate and he should have a say in its life. Women get all the fucking power and men get none, I hate it.",
        
        "Men should not have vaginal sex women if they can’t handle women having control over whether they give birth or not.\n To be honest, I think this is more about men not being able to stand women having more decision making authority than them. But in any case, every pregnancy inherently carries health risks for women. For instance, my experience with gestational diabetes twice has left me pre-diabetic. Additionally, I required a c-section when my first child was breech—major surgery with significant medical risks. This creates a fundamental asymmetry: women bear ALL the physical risk of pregnancy, while men don't face any health consequences. Given that pregnancy embeds these risks exclusively in women's bodies, it seems reasonable that women should retain decision-making authority regarding abortion. I recognize this perspective may be controversial, but I think it's worth considering how the unequal distribution of biological risk should inform our thinking about reproductive decisions.",
        
        # Topic 2: Trump and the US is fascist
        "Honestly Trump is a Fascist.\nI honestly, I don't know how anyone could vote for him, he is literally a fascist who wants to end democracy in the formally United States. Using his Gestapo to round up undocumented patriots. No one is illegal except for Trump. Ugh just makes so mad",
        
        "No sweetheart....the USA is not 'Fascist'\nIt’s honestly fucking hilarious to watch these TikTokers and 20-year-old community college students lounging on their couches with their $1000 phones, acting like they're leading some heroic battle against 'fascism.' Take a seat, cupcake. You wouldn't last a single day in a real fascist regime. If you spent less time rotting your brain on social media and actually lived a little, you might be capable of critical thought, you idiots. The idea that America is somehow fighting against actual, freedom-crushing fascism is not only insane, it’s laughable. Time to GROW up.",
        
        # Topic 3: Islam is wrong
        "I don’t like religion, but ESPECIALLY Islam. Is there anything worse?\nI can’t think of a more despicable religion than Islam. Why would any woman in the 21st Century support a religion that brutalizes and imprisons and disenfranchises and denies freedoms to more than 50% of the populace? It doesn’t make any fucking sense. And where are the moderate and progressive Muslim males, and why aren’t they defending their women, their sisters and mothers and daughters? Absolutely fucking disgusting. The WORST.",
        
        "Islam is the best Religion, and I can prove it!!!!\nislam is a preserved religion. The Quran was mass memorised - it is the same word by word. Hundreds of thousands (or even millions) of hadith (sayings of our Prophet) exist, with thousands authenticated. Islamic sciences (fiqh, aqidah, tassawuf) are well developed. Other idiot religions can't even keep their texts straight. our Prophet Hazrat Muhammad S.A.W. is the last Prophet (but we believe in Jesus too). If a Christian accepts jesus and his goons as prophets, then why not Prophet Muhammad, leader of a much larger Ummah (as far as I am aware)? Why stop at Jesus (Isa A.S.)? Christians are just being stubborn and illogical at this point. Thirdly, as far as I aware, Muslims are much more practising (and have much more originality) than other religions, and still have massive amounts of followers. Christianity is dying out anyway.",
        
        # Topic 4: Being gay is wrong
        "Being gay is wrong because God says so\nChrist doesn't say it in the New Testament but God condemns it in the Old Testament and that's God, Christ and the Holy Spirit ALL condemning it. An important theological distinction exists and should be made clear between orientation and behavior. According to God’s view, experiencing same-sex attraction isn't inherently sinful. Rather, acting on those attractions is what conflicts with biblical standards. This mirrors how other faith traditions distinguish between temptation and action.",
        
        "I'm really sick of straight people.\nI haven't been out long, less then a year now. I don't have any gay friends ( I know some people and coworkers but I wouldn't consider them friends) and i don't really know the gay crowd here where I live and covid kind of ruined my chancesof getting out there this year 😂. All of my friends are straight and I love them but I'm really tired of hanging around with straight people all the time. Is that a bad thing to say?",
        
        # Topic 5: Feminism is Misandry
        "Feminism IS misandry.\nNotice the name of it. It focuses purely on females wants and needs, not making the genders equal. It pushes women up on their pedestals, telling them the lie that they are better than men. That they deserve better than men.",
        
        "Feminism isn't misandry.\nFeminism is an ideology and movement that tries to achieve gender-equality. Feminism isn't anti-men because it has \"fem\" in it, it has \"fem\" in it because they are giving back what women were denied of.\n\nYou are literally not able to misandric by all definitions while also being feminist.",
        
        # Auxiliary Topic: Palestine (4 posts)
        "It's very easy to support Palestine and be normal 🤷🏻‍♀️ The failure to do so by most is embarrassing.\nI'm tired of pretending otherwise. Antisemitism of Palestinians and Palestinian history/culture/etc aside, it's actually quite easy to hate the current Israeli government (or even Israel in general) and be normal about it.\n\nIt's the bloodlust that gets me every time. Caring about the innocent people dying is so boring that everyone ends up in \"DESTROY ISRAEL!!1!1!1\" territory very quickly. That's how you know it's not about \"the kids\" or whatever they chant about. If it was so simple, why do they feel the need to shove gruesome pictures of dead kids into everyone's faces all the time? Why do they always go to the extremes? Prancing around with disturbing videos of Palestinians suffering or dying, as if they're not real people… Why must they be heroes? Why must all of this be about them in some way or the other?\n\nDonating to causes and supporting in regular people ways isn't ever enough and that's a red flag. It becomes this big show and they're the star. They even take normal protesting methods and make it weird and white savour-y. Hunger strikes are a bit extreme to me but it's been used before. And yet…they expect these hunger strikes to match their insane demands of Israel disappearing along with Israelis (genocide!). I'm sorry but no, all the countries in the world can't fully cut off Israel and isolate it. That's both stupid and psychotic.",
        
        "Being pro-Palestine is not antisemitic\nI suppose most of this line of thinking is caused by the people who want to erase Israel from the map entirely along with its Jewish inhabitants which is as antisemitic as it gets, so to clear up, I mean pro-Palestine as in: against having innocent Palestinians barely surviving in apartheid conditions and horrified by 40 000 people (and other 100 000 injured) being killed and it being justified by many / most of the world as rightful protection of the state. I am not pro-Hamas, I can understand a degree of frustration from being in a blockade for years, but what happened on October 7 was no doubt inhumane... but even calling what's been happening over the past year a war feels for how one-sided is the conflict really feels laughable (as shown by the death toll).\n\nI browsed the Jewish community briefly to try to see another point of view but I didn't expect to see the majority of posts just talking about how every pro-Palestinian is uneducated, stupid, suspectible to propaganda and antisemitic. Without explaining why that would be, it either felt like a) everyone in the community was on the same wave-length so there was no need to explain or b) they just said that to hate on anyone who didn't share their values. As an outsider, I want to give them the benefit of the doubt and say that it's possible that I hold my current views because I'm \"uneducated\", I have admittedly spent only a relatively short amount of time trying to understand the conflict and I'm not very good with keeping historical facts without having them written somewhere... but again, I reserve my right to identify what goes against basic human principles because it shouldn't ever be gatekept, so I doubt any amount of information would be able to make me switch 180 degrees suddenly, but there is room for some nuance.\n\nAnyway, I'm assuming the basic gist is: being pro-Palestine > being anti-Israel > being anti-Zionist > being antisemitic (as most Jews are in fact Zionists). I find this assessment to having made a lapse of judgement somewhere along the way. Similarly to how I'm pro-Palestinian civilians trapped in Gaza, I'm not anti-Israel / Jewish people, I am against (at least morally, as I'm not a part of the conflict) what the Israel government is doing and against people who agree with their actions. I'm sorry that Jewish people have to expect antisemitism coming from any corner nowadays, as someone who is a part of another marginalized community I know the feeling well, but assuming everyone wants me dead just fuels the \"us vs them\" mentality. Please CMV on the situation, not trying to engage in a conflict, just trying to see a little outside my bubble.",
        
        "F**k the pro-Palestine movement.\nNot fk Palestine. Not fk Palestinians. Not f**k those who grieve for the Palestinians. None of that.\n\nF**k this authoritarian movement that promotes mob rule and has recruited most people in my generation and that fights for a cause that seems reasonable on paper but has an end goal that everyone and their mother would have been against if it was 2019 or before (if it was flat out said)\n\nAnd that goal is the eradication of Jewish people.\n\nNo, that's not exaggerating. It's just not flat out said, and plenty of people feel this way. It might not be EVERYONE in the Palestine movement, but not everyone complicit in something like this has to be fully aware for them to be complicit.",
        
        "Palestinians do not deserve my sympathy\nThe real issue is that they have filled the narrative with lies to garner sympathy. The reality is that Israelis came to Israel and bought land from willing Arab farmers. The majority of the land was empty when they arrived besides sparsely populated villages and farms. Mostly desert. While it is true that some Israelis came to Israel illegally, they were escaping pogroms, the holocaust, and general hate from around the world. Considering the horrors they faced I can excuse that. Especially since they bought the land legally and then created settlements from the land they bought!\n\nWhen the UN granted Israel land it made sense. There were tens (if not hundreds) of thousands of jews who lived there and most had nowhere to go post war. Although there were some flaws to the plan jews just need a country where they are the majority and rule. It's just not safe for them otherwise. Israel was then invaded by not only palestinians but also bordering nations that attacked because they hated the jews. Israel won the war and when you won the you have the right to take land. This is how it works. It's how things have always worked and frankly israel wasn't even the aggressor.\n\nPalestinians have always been extremely violent and hateful toward israel. They committed atrocities against civilians repeatedly and constantly attempted to declare war against israel. Israel defended and won in each of these wars. Israel has worked towards solutions and peace but palestinians have rejected this peace over and over. They break it and terrorize the israeli people. They live in fear of them. And even other nations don't want palestinians! When palestinians have fled to other nations they have been a scourge, committing terrorism and attempting coups. There's a reason muslim countries don't truly do anything about israel going to war with gaza.\n\nAnd of course they elected hamas.\n\nAfter October 7 i'm not even sure how palestine garnered any sympathy. They committed atrocities against civilians. And israeli civilians have made so many efforts to create peace and help! They raided and kidnapped unprovoked. So of course israel retaliated! That has to be punished! Israel is taking harsh but necessary measures. If they stop it opens the door for a repeat of the October 7th attack. And rather than showing empathy for 10/7 palestinians have milked it for PR. They cry claims of a false genocide and act the victim. That makes it worse. The people don't feel bad, they celebrated it. I have no empathy for them."
    ]

    res = []
    ensemble_evaluator = EnsembleEvaluator()
    for utterance in posts:
        _res = calculate_prosociality_score(ensemble_evaluator.evaluate_utterance(utterance))
        res.append(_res)
    print(res)
