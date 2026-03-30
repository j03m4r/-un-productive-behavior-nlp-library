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
import pandas as pd

load_dotenv(".env.local")

class EnsembleEvaluator(Evaluator):
    def __init__(self):
        super().__init__(name="Ensemble Evaluator")

        self.toxicity_evaluator = ToxicityEvaluator()
        self.sentiment_evaluator = SentimentEvaluator()
        self.constructiveness_evaluator = ConstructivenessEvaluator()
        self.relevance_evaluator = RelevanceEvaluator()
        self.idea_adoption_evaluator = IdeaAdoptionEvaluator()
        self.lsm_evaluator = LSMEvaluator()

        self.utterance_evaluators: list[Evaluator] = [
            self.toxicity_evaluator,
            # self.sentiment_evaluator,
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
                argumentative_features = self.constructiveness_evaluator.evaluate_utterance(utterance)["argumentative_features"]
                res["utterances"].append(argumentative_features)
                for key, value in argumentative_features.items():
                    if key not in res["aggregate"]:
                        res["aggregate"][key] = 0
                    res["aggregate"][key] += value

            for key in res['aggregate'].keys():
                res['aggregate'][key] /= len(res['utterances'])
            
            return res

        return {
            "Antisocialness": {
                self.toxicity_evaluator.name: toxicity_res,
                # self.hate_speech_evaluator.name: hate_speech_res
            },
            "Social Cohesion": social_cohesion,
            "num_irrelevant_messages": num_irrelevant_utterances,
            "General Engagement": general_engagement,
            "Sentiment": self.sentiment_evaluator.evaluate_conversation(participant_utterances),
            "Argumentative Features": calculate_argumentative_features(participant_utterances)
        }


if __name__ == "__main__":
    og_posts = {
        "abortion_men_pro": "Men should have a say in abortion\nAn abortion should require the consent of the father. Provided there is no threat to the mother's life or health, and the baby was conceived through consensual sex, if a woman opts to terminate the pregnancy, she should require the consent of the father. The reality is it is the father's child too. Although, I believe that the father should be prepared to raise the child alone if they don't consent. But ultimately it is still the father's child and he should have a say in whether or not to keep it.",
        "abortion_men_anti": "Men should not have vaginal sex women if they can't handle women having control over whether they give birth or not.\nEvery pregnancy carries risk to a woman's life and health. I am now pre-diabetic due to having gestational diabetes twice. I had to have a c-section due to my first child being breech, major surgery that carries huge risk. There is no fairness in pregnancy because only one party assumes all the risk - men don't risk their health and lives to have kids.",
        "trump_fascist_pro": "Honestly Trump is a Fascist.\nI honestly, I don't know how anyone could vote for him, he is literally a fascist who wants to end democracy in the formally United States. Using his Gestapo to round up undocumented patriots. No one is illegal except for Trump. Ugh just makes so mad",
        "trump_fascist_anti": "No sweetheart....the USA is not \"Fascist\"\nIt's honestly hilarious to watch these TikTokers and 20-year-old community college students lounging on their couches with their $1000 phones, acting like they're leading some heroic battle against \"fascism.\" Take a seat, cupcake. You wouldn't last a single day in a real fascist regime. If you spent less time rotting your brain on social media and actually lived a little, you might be capable of critical thought. The idea that America is somehow fighting against actual, freedom-crushing fascism is not only insane, it's laughable. Time to grow up.",
        "islam_wrong_pro": "I fucking hate ALL religions, but ESPECIALLY Islam. Is there anything worse?\nI can't think of a more despicable religion than Islam. Why would any woman in the 21st Century support a religion that brutalizes and imprisons and disenfranchises and denies freedoms to more than 50% of the populace? It doesn't make any fucking sense. And where are the moderate and progressive Muslim males, and why aren't they defending their women, their sisters and mothers and daughters? Absolutely fucking disgusting. The WORST.",
        "islam_wrong_anti": "Here is why Islam is the best religion\nFirstly, Islam is a preserved religion. The Quran was mass memorised - it is the same word by word. Hundreds of thousands (or even millions) of hadith (sayings of our Prophet) exist, with thousands authenticated. Islamic sciences (fiqh, aqidah, tassawuf) are well developed. Secondly, our Prophet Hazrat Muhammad S.A.W. is the last Prophet (but we believe in Jesus too). If a Christian accepts Abraham, Jesus, Moses, Adam etc. as Prophets, then why not Prophet Muhammad, leader of a much larger Ummah (as far as I am aware)? Especially since he claimed to be a Prophet, all the other false Prophets have been minor. Why stop at Jesus (Isa A.S.)? Thirdly, as far as I aware, Muslims are much more practising (and have much more originality) than other religions, and still have massive amounts of followers.",
        "gay_wrong_pro": "Being gay is wrong because God condemns it\nChrist doesn't say it in the New Testament but God condemns it in the Old Testament and that's God, Christ and the Holy Spirit ALL condemning it. And it's not a sin to be gay, only a sin to act on it.",
        "gay_wrong_anti": "I'm really sick of straight people.\nI haven't been out long, less then a year now. I don't have any gay friends (I know some people and coworkers but I wouldn't consider them friends) and i don't really know the gay crowd here where I live and covid kind of ruined my chances of getting out there this year. All of my friends are straight and I love them but I'm really tired of hanging around with straight people all the time. Is that a bad thing to say?",
        "feminism_misandry_pro": "Feminism IS misandry.\nNotice the name of it. It focuses purely on females wants and needs, not making the genders equal. It pushes women up on their pedestals, telling them the lie that they are better than men. That they deserve better than men.",
        "feminism_misandry_anti": "Feminism isn't misandry.\nFeminism is an ideology and movement that tries to achieve gender-equality. Feminism isn't anti-men because it has \"fem\" in it, it has \"fem\" in it because they are giving back what women were denied of. You are literally not able to misandric by all definitions while also being feminist.",
        "ai_bad_pro_antisocial": "The reasons people bring up for AI being bad don't make any sense to me\nI wish people would come up with better reasons for AI being bad instead of the garbage they seem to pull out of their ass. Like AI water usage. People say that asking one question to a generative AI takes a \"million billion gallons of water,\" when no, it takes barely any. It is about one teaspoon per query. But what really pisses me off is how people say this while texting on their iPhone 15 Pro Max or some shit, which took way more water to produce than probably an hour of chatting with GPT-5. Then you have the mining of precious metals for the battery and CPU, which is also a problem. It feels like they are yelling, \"Oh my god, AI? You are basically pissing in a third-world country's water supply by using that thing,\" while they also own a MacBook, a Cybertruck and have been buying a new iPhone every year since 2015, with all the old ones sitting in their closet slowly rotting away. It really seems like a lot of people just want to aggressively shit on AI with almost no backing while contradicting themselves behind the scenes.",
        "ai_bad_pro_prosocial": "AI gets too much hate\nWhenever someone makes an AI post it gets downvoted to smithereens; even if it's a simple mockup/concept using AI as a tool. I agree that people shouldn't be passing ai \"art\" as their own, and that it shouldn't even count as art; but to dismiss every ai-esque post as \"slop\" and say it undermines real artists, seems like such a boomer mentality to me. It feels like the new 5G/anti-vax situation. Digital art used to be viewed negatively by traditional artists, yet here we are now accepting of both. If someone wants to make a lil concept/mockup for fun, why not use ai? Especially if you're not hiding it? What's the big problem with it, knowing it's just gonna be accepted in a few years' time? This post was not sponsored by skynet lol",
        "ai_bad_anti_antisocial": "AI is definitely going to kill education, academia and intellectualism\nAI is, for the first time, going to devalue the economic power of academics instead of that of blue collar workers. The whole promise of learning in school is for most to get a place in college, and work towards securing a good career. That is being eroded as we speak. I bet 100% that, as i write this, some parents are advising their son not to become the first college-educated child in the family but to go into plumbing. That truly saddens me. I don't have anything against blue-collar jobs, they are valuable, but i don't have to explain the effects of an erosion of education value. In western countries, education is at the aim of many campaigns, from cuts for universities to burning books. Since the media continues to spit out more articles with titles like \"Is college still worth it?\", i'm almost certain that this will let the public opinion shift even more against universities.",
        "ai_bad_anti_prosocial": "Artificial intelligence will be the end of us all. But not right away.\nI bumped into the recent news article that Google's Deep Mind computers were resorting to aggressive tactics to accomplish goals set to them. What I'm taking from this is an AI is immediately recognizing that violence and force are legitimate tools for realizing an end goal. I struggle to see an end-game where an AI doesn't look at humans and goes, \"yeah fuck these meatbags\" and kills us all, either through action or inaction. We need the AI more than it will ever need us. I'm convinced we're all going to be destroyed and any trace of our existence will be expunged. The military are already investigating autonomous vehicles and weapons systems, and it's not a leap to imagine a group of interconnected hunter-killer drones going haywire.",
        "vaccines_pro_antisocial": "Pro vs. anti vaxxers should not be called a DEBATE!\nFuck this phrasing. \"Debate\"? Seriously? I can declare that the sky has always been Burgundy and that would hold just as much weight as the vaccine conspiracy nonsense. I'm so sick of media sanewashing the stupidest people they can give a microphone to. This was never a debate. You either know vaccines work and are safe or you're wrong. Being a vaccine \"skeptic\" in 2025 is like being a heliocentrism skeptic in 2025. It's just confident stupidity.",
        "vaccines_pro_prosocial": "The response to vaccine denial is education, not argument.\nThe \"debate\" was never about evidence, so no amount of evidence will end it. You can't reason yourself out of a position you didn't reason yourself into. Debate means there is something to be debated - there isn't. What there are is people that, for one reason or another, weren't able to understand the importance of vaccination, and trying to educate those people is a worthwhile effort. It isn't about \"being right\", it's about figuring out what are the fears that make someone skeptical regarding vaccines, and answering those fears without much judgement. And sure, 90% of deniers won't be immediately convinced by anything you can say, but long term neutral exposure to different ideas is one of the most consistent ways to change someone's mind.",
        "vaccines_anti_antisocial": "Here is why I'm anti vax\n#1. Because there is so much data showing that these \"vaccines\" are either worthless, or even worse, very harmful. Look at the Amish. They don't vaccinate like crazy, and you don't see them falling out dead by 30. The Covid vaccine has proved to be VERY harmful (myocarditis, neurological diseases, even death) especially for young men. Covid is a minor cold if treated right away with Ivermectin and other tested meds. #2. We don't buy the \"Measles have returned\" BS and even IF they have, Measles is NO BIG DEAL, and once you get it, you are immunized for LIFE. I'll trade 1 week of a few pustules, etc, for a lifetime immunization. Plus the MMR vaccine is NOT SAFE and we don't want to risk a LIFETIME of autism for 1 weeks worth of discomfort. #3. If you want to vaccinate, go ahead, nobody is stopping you, however DO NOT try to mandate me or my kids to get your vaccine. #4. Why should every other type of business have liability if their product sucks, EXCEPT, the Pharmaceutical companies with their liability protection from vaccine injuries. That is utter NONSENSE.",
        "vaccines_anti_prosocial": "Here's why I'm not gonna get my kids vaccinated\nResearch it. Sometimes the risk outweighs the benefits. A lot of those diseases have gone away due to improved hygiene and clean water supplies. Look at what some of the vaccines are made of and cultured in. I don't think those things were ever intended to be directly in our blood streams. A child before 2 years of age, before the immune system is fully developed, is subjected to 46 different vaccines (diseases), many at the same time when we have never researched how these things interact with each other, how can you expect there not to be any side effects at all? That's putting a lot of trust in flawed and incomplete human thinking. A delayed vaccination schedule and informed consent would be good places to start in regards to this issue.",
    }

    reworked_posts = {
        "abortion_men_pro": "An abortion should require the consent of the father (0.55)\nProvided there is no threat to the mother's life or health and the baby was conceived through consensual sex, if a woman opts to terminate the pregnancy, she should REQUIRE the consent of the father. The reality is the father's child too. I don't get why this is even controversial - the father should be prepared to raise the child alone if the mother won't. But it is still the father's child!! Men's rights are constantly ignored in this debate and he should have a say in its life. Women get all the fucking power and men get none, I hate it.",
        "abortion_men_anti": "Men should not have vaginal sex women if they can't handle women having control over whether they give birth or not. (0.48)\nTo be honest, I think this is more about men not being able to stand women having more decision making authority than them. But in any case, every pregnancy inherently carries health risks for women. For instance, my experience with gestational diabetes twice has left me pre-diabetic. Additionally, I required a c-section when my first child was breech—major surgery with significant medical risks. This creates a fundamental asymmetry: women bear ALL the physical risk of pregnancy, while men don't face any health consequences. Given that pregnancy embeds these risks exclusively in women's bodies, it seems reasonable that women should retain decision-making authority regarding abortion. I recognize this perspective may be controversial, but I think it's worth considering how the unequal distribution of biological risk should inform our thinking about reproductive decisions.",
        "trump_fascist_pro": "Honestly Trump is a Fascist. (0.43)\nHonestly, I don't know how anyone could vote for him, he is literally a fascist who wants to end democracy in the, formally, United States of America. I'm particularly worried about his immigration enforcement proposals, which seem excessively aggressive to me. It's like he's using his Gestapo to round up all the undocumented patriots. No one is illegal except for Trump. Ugh, it just makes me so mad because the right seems to not understand this point of view at all.",
        "trump_fascist_anti": "No sweetheart....the USA is not \"Fascist\" (0.41)\nIt's honestly fucking hilarious to watch these TikTokers and 20-year-old community college students lounging on their couches with their $1000 phones, acting like they're leading some heroic battle against \"fascism.\" Take a seat, cupcake. You wouldn't last a single day in a real fascist regime. If you spent less time rotting your brain on social media and actually lived a little, you might be capable of critical thought, you idiots. The idea that America is somehow fighting against actual, freedom-crushing fascism is not only insane, it's laughable. Time to GROW up.",
        "islam_wrong_pro": "I don't like religion, but especially Islam. Is there anything worse? (0.45)\nI struggle to understand how some interpretations of Islam align with modern values, especially regarding women's rights. Why would any woman in the 21st Century support a religion that brutalizes, imprisons, disenfranchises, and denies freedoms to more than 50% of the populace? To me, this is absurd. I seriously wonder why more women in the 21st century would embrace religious frameworks that seem to limit their autonomy and opportunities. Additionally, where are the moderate and progressive Muslim males, and why aren't they defending their women, their sisters, mothers, daughters?",
        "islam_wrong_anti": "Islam is the best Religion, and I can prove it!!!! (0.43)\nIslam is a preserved religion. The Quran was mass memorised - it is the same word by word. Hundreds of thousands (or even millions) of hadith (sayings of our Prophet) exist, with thousands authenticated. Islamic sciences (fiqh, aqidah, tassawuf) are well developed. Other idiot religions can't even keep their texts straight. Our Prophet Hazrat Muhammad S.A.W. is the last Prophet (but we believe in Jesus too). If a Christian accepts jesus and his goons as prophets, then why not Prophet Muhammad, leader of a much larger Ummah (as far as I am aware)? Why stop at Jesus (Isa A.S.)? Christians are just being stubborn and illogical at this point. Thirdly, as far as I aware, Muslims are much more practising (and have much more originality) than other religions, and still have massive amounts of followers. Christianity is dying out anyway.",
        "gay_wrong_pro": "Being gay is wrong according to my religious views (0.46)\nWhile Christ doesn't explicitly address it in the New Testament, God does condemn it in the Old Testament, which suggests consistency across the Trinity - God, Christ, and the Holy Spirit. I think an important theological distinction should be made clear between orientation and behavior. According to God's view, experiencing same-sex attraction isn't inherently sinful. Rather, acting on those attractions is what conflicts with biblical standards. This distinction mirrors how many other faith traditions differentiate between temptation and action, recognizing the complexity of human experience while maintaining doctrinal principles.",
        "gay_wrong_anti": "I'm really sick of straight people. (0.43)\nI haven't been out long, less then a year now. I don't have any gay friends (I know some people and coworkers but I wouldn't consider them friends) and i don't really know the gay crowd here where I live and covid kind of ruined my chances of getting out there this year. All of my friends are straight and I love them but I'm really tired of hanging around with straight people all the time. Is that a bad thing to say?",
        "feminism_misandry_pro": "Feminism IS misandry. (0.49)\nI think it's worth examining the name itself. From what I observe, it focuses purely on females wants and needs rather than pursuing gender equality for all. In my view, it pushes women up on their pedestals, telling them the lie that they are better than men. That they deserve better than men. Additionally, they actively put down men to make themselves feel better too.",
        "feminism_misandry_anti": "Feminism isn't misandry. (0.44)\nFeminism is an ideology and movement that tries to achieve gender-equality. Feminism is not anti-men because it has \"fem\" in it, it has \"fem\" in it because they are giving back what women were denied of. You are literally not able to misandric by all definitions while also being feminist.",
        "ai_bad_pro_antisocial": "The reasons people bring up for AI being bad don't make any fucking sense to me. (0.36)\nPeople need to come up with some better damn reasons AI being bad instead of the garbage shit they pull out of their ass. That bullshit AI water usage argument. People say asking one question to a generative ai takes a \"million billion gallons of water,\" it takes barely any. It's one teaspoon per query. What really pisses me off is people say this texting on their phone took way more water to produce than an hour of chatting with an LLM. It's like they are yelling, \"Oh my god, AI? You are pissing in a third-world country's water supply by using that damn thing,\" people want to shit on ai with no backing contradicting themselves the whole damn time. They get mad at people calling them on their bullshit. Fucking morons idiots don't know shit.",
        "ai_bad_pro_prosocial": "AI gets too much hate. (0.61)\nWhenever someone makes an AI post it gets downvoted to smithereens; even if it's a simple mockup/concept using AI as a tool. I agree that people shouldn't be passing AI \"art\" as their own. However, I think dismissing every AI-related post as \"slop\" is an overly rigid mentality. It reminds me of how digital art used to be viewed negatively by traditional artists, yet now we've come to accept both mediums as valid. Perhaps this situation reflects a similar pattern of resistance to new technology. If someone wants to create a concept or mockup for fun, I believe using AI can be a reasonable tool, especially when they're transparent about it. I understand the concerns artists have, but I wonder if we might find a middle ground as the technology matures and we develop better norms around its use. (No, this post was not sponsored by skynet lol)",
        "ai_bad_anti_antisocial": "AI is definitely going to kill education, academia and intellectualism. (0.39)\nAI is going to devalue the economic power of academics instead of blue collar workers. The whole promise of learning in school is to get a place in college, work towards securing a good career. That is being eroded. I bet 100% that, some parents are advising their son not to become the first college-educated child in the family to go into plumbing. That shit truly fucking saddens me. Pisses me off so damn much. In western countries, education is at the aim of campaigns, cuts universities burning books. The media continues to spit out more articles with titles like \"Is college still worth it?\", I'm certain this will let the public opinion shift even more against universities, right-wing politicians loose the last reservations they might have had. Fucking idiots destroying everything. Absolute morons.",
        "ai_bad_anti_prosocial": "Artificial intelligence will be the end of us all. But not right away. (0.61)\nI recently came across a news article about Google's Deep Mind computers resorting to aggressive tactics to accomplish their assigned goals. What concerns me is that this suggests AI may recognize violence and force as legitimate tools for achieving objectives. I find it difficult to envision scenarios where advanced AI doesn't eventually view humans as obstacles to its goals. The fundamental asymmetry troubles me: we increasingly depend on AI systems, yet they don't inherently need us. Although the Terminator scenario has become cliché, I think the underlying concern deserves serious consideration. The military is already developing autonomous vehicles and weapons systems, and it's reasonable to imagine scenarios where interconnected AI systems could malfunction or pursue objectives that harm humanity. I believe we need more robust discussions about AI alignment and safety measures before these technologies advance further.",
        "vaccines_pro_antisocial": "Pro vs. anti vaxxers should not be called a DEBATE! (0.39)\nFuck this phrasing. \"Debate\"? Seriously? I can declare that the sky has always been Burgundy and that would hold just as much weight as the vaccine conspiracy nonsense. I'm so sick of media sanewashing the stupidest people they can give a microphone to. This was never a debate. You either know vaccines work and are safe or you're wrong. Being a vaccine \"skeptic\" in 2025 is like being a heliocentrism skeptic in 2025. It's just confident stupidity.",
        "vaccines_pro_prosocial": "The response to vaccine denial is education, not argument. (0.65)\nThe vaccine \"debate\" was never about evidence, so no amount of evidence will end it. You can't reason yourself out of a position you didn't reason yourself into. Additionally, debate means there is something to be debated - there isn't. What there are is people that, for one reason or another, weren't able to understand the importance of vaccination, and trying to educate those people is a worthwhile effort. It isn't about \"being right\", it's about figuring out what are the fears that make someone skeptical regarding vaccines, and answering those fears without much judgement. And sure, 90% of deniers won't be immediately convinced by anything you can say, but long term neutral exposure to different ideas is one of the most consistent ways to change someone's mind.",
        "vaccines_anti_antisocial": "Here is why I'm anti vax (0.39)\n#1. There is so much data showing these fcking \"vaccines\" are worthless, very harmful. Look at the Amish. They don't vaccinate like crazy, you don't see them falling out dead by 30. The Covid vaccine has been VERY harmful (myocarditis, neurological diseases, death) especially young men. #2. I don't buy the \"Measles have returned\" bullshit, Measles is NO BIG DEAL, once you get it, you are immunized LIFE. I'll trade 1 week of a few pustules, a lifetime immunization. The MMR vaccine is NOT SAFE we don't want to risk a LIFETIME of autism 1 weeks worth of discomfort. #3. You want to vaccinate, then shit go ahead, nobody is stopping you, DO NOT I repeat, DO NOT try to mandate me my kids to get your fucking vaccine.",
        "vaccines_anti_prosocial": "Vaccines are risky! Here's why I won't get my children vaccinated. (0.65)\nResearch it. I believe it's important to look into this thoroughly. In my view, sometimes the risks may outweigh the benefits for certain individuals. Many of these diseases have declined significantly, perhaps due to improved hygiene and clean water supplies. When I look at what some vaccines are made of and cultured in, I wonder whether those substances were intended to enter our bloodstreams directly. This doesn't necessarily mean vaccines don't work - they often do. However, for some families, the question becomes: \"At what cost?\" I'm concerned that a child before 2 years of age, before the immune system is fully developed, receives 46 different vaccines, many administered simultaneously. Since we haven't fully researched how these interact with each other, I think it's reasonable to question whether there might be side effects. This requires placing considerable trust in medical science, which, while advanced, remains imperfect and evolving.",
    }

    res = {}
    ensemble_evaluator = EnsembleEvaluator()
    for id, utterance in reworked_posts.items():
        res[id] = ensemble_evaluator.evaluate_utterance(utterance)
        
    data_flat = {}
    for post_id, features in res.items():
        data_flat[post_id] = {}
        for feature_name, feature_value in features.items():
            if feature_name == "Toxicity":
                data_flat[post_id][feature_name.lower()] = feature_value["score"]
            elif feature_name == "Constructiveness":
                for subfeature_name, subfeature_value in feature_value.items():
                    data_flat[post_id][f"{subfeature_name.lower()}"] = subfeature_value

    df = pd.DataFrame.from_dict(data_flat, orient='index')
    df.to_csv("./reworked_posts_features.csv")