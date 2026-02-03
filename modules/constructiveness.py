from modules import Evaluator
from textstat import smog_index
import json, subprocess
from pathlib import Path
import spacy

class ConstructivenessEvaluator(Evaluator):
    discourse_connectives = {
        "and", "or", "for", "as", "if", "so", "when", "but",
        "one", "then", "now", "after", "before", "here", "never", "since", 
        "always", "during", "later", "often", "once", "sometimes", "soon", 
        "usually", "earlier", "recently", "quickly", "first", "suddenly", 
        "immediately", "shortly", "formerly", "frequently", "occasionally", 
        "rarely", "seldom", "nowadays", "eventually", "ultimately", "presently",
        "directly", "initially", "originally", "subsequently", "currently",
        "momentarily", "lately", "continually", "constantly", "regularly",
        "intermittently", "temporarily", "permanently", "instantly", "promptly",
        "all of the time", "for much of the time", "in a year's time", 
        "it was at this time", "for long periods of time", "in the fullness of time",
        "in the same time", "it is about time", "after a short time", "to top it all",
        "time and time again", "there was a time", "now and again", "and all the time",
        "following", "until", "finally", "previously", "next", "last", "lastly",
        "in a very short time", "in a fortnight's time", "last of all", "time to move on",
        "by the by", "as a final point", "as of yet", "afore", "not long ago",
        "second", "third", "firstly", "secondly", "thirdly",
        "in the first place", "in the second place", "in the next place",
        "for another thing", "afterward", "the latter", "the former", 
        "second of all",
        "because", "because of", "due to", "owing to the fact that",
        "one effect of", "one result of",
        "whether", "unless", "even if", "provided that", "on condition that",
        "supposing that", "in case",
        "therefore", "thus", "hence", "consequently", "accordingly",
        "as a result", "as a consequence", "for this reason", "it follows that",
        "so that", "in that case", "under those circumstances", "in such an event",
        "arising out of this", "as a corollary", "apropos", "whereby", "thereby",
        "in a nutshell",
        "however", "though", "although", "yet", "nonetheless", "nevertheless",
        "while", "only", "anyway", "still", "even so", "all the same",
        "at the same time", "in any case", "in any event", "after all",
        "for all that", "in spite of that", "at any rate", "anyhow",
        "another distinction", "else",
        "rather", "instead", "alternatively", "conversely", "on the contrary",
        "on the other hand", "by contrast", "in contrast", "by comparison",
        "in comparison", "by way of contrast", "by way of comparison",
        "a striking difference", "contrariwise", "oppositely",
        "again", "at least",
        "more", "much", "also", "too", "further", "as well", "nor",
        "increasingly", "much more", "in addition", "in particular",
        "what is more", "moreover", "furthermore", "besides", "additionally",
        "apart from that", "aside from this",
        "piece by piece", "cap it", "crown it all", "top it off", 
        "on top of it all", "another interesting aspect",
        "like", "exactly", "similarly", "likewise", "equally",
        "in the same way", "by the same token", "along the same lines",
        "in a similar fashion", "in relation to", "with respect to",
        "with reference to", "in terms of", "in comparison with",
        "in comparison to", "compared with", "compared to", "by comparison with",
        "a different aspect", "ca.", "separate and distinct from",
        "incongruously",
        "such as", "for example", "for instance", "e.g.", "i.e.", "say",
        "namely", "more specifically", "to illustrate",
        "as an illustration", "as an illustration of", "as illustrated by",
        "as revealed by", "the previous example", "the example above",
        "the example below", "a rare example of", "a notable example of",
        "particularly", "that is", "in other words", "to put it another way",
        "to put it differently", "or rather", "better", "more accurately",
        "more precisely", "be exact", "put it mildly", "put it bluntly", 
        "what I mean is",
        "even", "indeed", "certainly", "especially", "the fact that",
        "stress", "emphasize", "significantly", "notably",
        "specifically", "above all", "more importantly", "most importantly",
        "chiefly", "mainly", "primarily", "in the main", "essentially",
        "in essence", "basically", "fundamentally",
        "by all means", "add to that", "regarding", "as to",
        "of critical importance", "of limited value", "in such a manner as",
        "of little significance", "of considerable significance", "of some value",
        "of doubtful value", "of equal weight", "of less significance",
        "of major interest", "of most value", "of overriding importance",
        "of profound importance", "of real value", "of strategic importance",
        "bar none", "the chief characteristics", "the major point",
        "the main issue is", "the most necessary", "of primary concern",
        "right", "well", "you know", "of course", "actually", "I mean",
        "you see", "naturally", "obviously", "clearly", "evidently",
        "manifestly", "undoubtedly", "undeniably", "admittedly", "surely",
        "in fact", "as a matter of fact", "in actual fact",
        "in reality", "in truth", "truly", "really", "according to",
        "it is evident from", "in every aspect", "to be frank", "supposedly",
        "technically speaking", "to be truthful", "what I am saying is",
        "put simply", "OK", "okay", "oh", "mind you", "as I was saying",
        "hypothetically", "comparatively speaking", "practically speaking",
        "academically", "it is an undeniable fact that",
        "generally", "on the whole", "in general", "as a rule",
        "by and large", "for the most part", "in most cases", "in most instances",
        "typically", "normally", "ordinarily", "commonly",
        "as a general rule", "in the majority of cases", "more often than not",
        "in neither case", "with all due respect", "one particular feature",
        "without a shadow of doubt", "the most worrying aspect",
        "the preferred mode of", "the principal characteristics", "up until now",
        "in conclusion", "to conclude", "to sum up",
        "to summarize", "in summary", "in sum", "in brief", "in short",
        "all in all", "overall", "in all", "taking everything into account",
        "taking everything together", "altogether", "resume", "get back to the point"
    }
    stance_adverbials = {
        "no doubt", "certainly", "undoubtedly", "arguably", "decidedly", 
        "definitely", "incontestably", "incontrovertibly", "of course",
        "probably", "perhaps", "maybe", "most likely", "very likely", 
        "quite likely", "I guess", "I think",
        "in fact", "really", "actually", "in actual fact", "for a fact", "truly",
        "evidently", "apparently", "reportedly", "reputedly",
        "mainly", "typically", "generally", "largely", "in general", "in most cases",
        "in our view", "from our perspective", "in my opinion",
        "like", "literally", "sort of", "about", "kind of", "roughly", 
        "so to speak", "if you can call it that",
        "as might be expected", "inevitably", "as you might guess", "to my surprise",
        "astonishingly", "surprisingly", "predictably",
        "unfortunately", "fortunately", "conveniently", "wisely", "sensibly",
        "quite rightly", "even worse", "disturbingly", "ironically",
        "most surprising of all", "even more importantly", "literally", "seriously", "honestly"
    }
    reasoning_lemmas = {
        "cause", "lead", "result", "produce", "create", "generate",
        "bring", "yield", "trigger", "induce", "prompt", "contribute",
        "prove", "demonstrate", "show", "indicate", "suggest", "imply",
        "reveal", "illustrate", "establish", "confirm", "verify",
        "substantiate", "corroborate", "evidence",
        "conclude", "infer", "deduce", "reason", "determine", "follow", "entail",
        "argue", "claim", "assert", "contend", "maintain", "hold",
        "posit", "propose", "advance",
        "explain", "account", "justify", "rationalize", "clarify", "elucidate",
        "mean", "signify", "ensure", "guarantee", "necessitate", "require", "dictate"
    }
    root_clause_verbs = {
        "think", "believe", "suppose", "guess", "assume", 
        "presume", "imagine", "reckon", "suspect", "expect",
        "feel", "sense", "find", "see", "notice", "observe",
        "consider", "regard", "judge", "deem", "hold",
        "know", "understand", "realize", "recognize", "acknowledge",
        "say", "claim", "argue", "maintain", "contend", "assert",
        "doubt", "wonder", "question"
    }

    def __init__(self):
        super().__init__(name="Constructiveness")

    def evaluate_utterance(self, text: str) -> dict:
        word_count = len(text.split())
        readability = smog_index(text)

        # Computing politeness features with R script (annoying)
        r_script = Path(__file__).resolve().parent.parent / "r" / "politeness_eval.R"
        cmd = ["Rscript", str(r_script), text]
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

        # Counting named entities
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        num_entities = len(doc.ents)

        # Counting argumentative features
        argumentative_features = {
            "discourse_connectives": 0,
            "stance_adverbials": 0,
            "reasoning_verbs": 0,
            "modals": 0,
            "full_root_clauses": 0,
            "partial_root_clauses": 0
        }
        text = text.lower()
        for dc in self.discourse_connectives:
            if dc in text:
                argumentative_features["discourse_connectives"] = argumentative_features.get("discourse_connectives", 0) + 1
        
        for stance_adverbial in self.stance_adverbials:
            if stance_adverbial in text:
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

        return {
            "word_count": word_count,
            "readability": readability,
            "politeness": politeness_dict,
            "named_entities": num_entities,
            "argumentative_features": argumentative_features
        }
