from modules import Evaluator
import spacy
import re

class LSMEvaluator(Evaluator):
    def __init__(self):
        super().__init__(name="LSM")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Use sets for O(1) lookup and remove overlaps
        self.personal_pronouns = {"i", "me", "you", "he", "she", "they", "it", 
                                  "him", "her", "them", "we", "us"}
        self.impersonal_pronouns = {"one"}
        self.articles = {"a", "an", "the"}
        self.prepositions = {
            "aboard", "about", "above", "absent", "across", "after", "against", "along", 
            "alongside", "amid", "amidst", "among", "amongst", "around", "as", "at", "atop",
            "bar", "barring", "before", "behind", "below", "beneath", "beside", "besides", 
            "between", "beyond", "but", "by",
            "circa", "concerning", "counting",
            "despite", "down", "during",
            "effective", "except", "excepting", "excluding",
            "failing", "following", "for", "from",
            "in", "including", "inside", "into",
            "less", "like",
            "minus",
            "near", "notwithstanding",
            "of", "off", "on", "onto", "opposite", "out", "outside", "over",
            "past", "pending", "per", "plus",
            "regarding", "respecting",
            "short", "since",
            "than", "through", "throughout", "to", "toward", "towards",
            "under", "underneath", "unlike", "until", "up", "upon",
            "versus", "vs.", "vs", "via",
            "wanting", "with", "within", "without", "worth"
        }
        self.auxiliary_verbs = {
            "best", "better", "can", "could", "dare", "may", "might", "must",
            "need", "ought", "shall", "should", "will", "would", "be",
            "do", "does", "did", "have", "has", "had",
            "am", "is", "are", "was", "were", "been", "being"
        }
        self.frequency_adverbs = {
            "always", "annually", "constantly", "continually", "continuously",
            "daily", "eventually", "ever", "frequently", "generally",
            "hourly", "infrequently", "intermittently", "later", "monthly",
            "never", "nightly", "normally", "now", "occasionally",
            "often", "periodically", "quarterly", "regularly",
            "scarcely", "seldom", "sometimes", "soon", "then",
            "today", "tonight", "usually", "weekly", "yearly",
            "yesterday", "yet"
        }
        self.negations = {
            "not", "nor", "no", "nowhere",
            "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", 
            "hadn't", "won't", "wouldn't", "don't", "doesn't", "didn't", 
            "can't", "couldn't", "shouldn't", "mightn't", "mustn't", "shan't"
        }
        self.quantifiers = {
            "all", "every", "each", "everything", "everybody", "everyone",
            "most", "many", "much", "lots", "plenty", "numerous", "countless",
            "loads", "tons", "heaps", "some", "several", "various", "certain",
            "few", "little", "barely", "hardly", "none", "nothing", "nobody",
            "neither", "any", "anything", "anybody", "anyone", "either",
            "more", "less", "fewer", "least", "fewest", "enough", "sufficient",
            "both", "half", "double", "twice"
        }

    def _count_word_occurrences(self, text, word_set):
        """Count whole word matches only, not substrings"""
        count = 0
        words = text.lower().split()
        for word in words:
            # Remove punctuation from word
            clean_word = re.sub(r'[^\w\']', '', word)
            if clean_word in word_set:
                count += 1
        return count

    def _count_conjunctions(self, text):
        """Count conjunctions using spaCy POS tags"""
        doc = self.nlp(text)
        count = 0
        for token in doc:
            if token.pos_ == "CCONJ" or token.pos_ == "SCONJ":
                count += 1
        return count

    def evaluate_conversation(self, conversation: list[str]) -> dict:
        p1_text = " ".join([utt for idx, utt in enumerate(conversation) if idx % 2 == 0])
        p2_text = " ".join([utt for idx, utt in enumerate(conversation) if idx % 2 == 1])
        
        p1_word_count = len(p1_text.split())
        p2_word_count = len(p2_text.split())
        
        if p1_word_count == 0 or p2_word_count == 0:
            return {"avg_lsm_score": 0.0}
        
        categories = {
            "personal_pronouns": self.personal_pronouns,
            "impersonal_pronouns": self.impersonal_pronouns,
            "articles": self.articles,
            "prepositions": self.prepositions,
            "auxiliary_verbs": self.auxiliary_verbs,
            "frequency_adverbs": self.frequency_adverbs,
            "negations": self.negations,
            "quantifiers": self.quantifiers
        }
        
        p1_counts = {}
        p2_counts = {}
        
        for category, word_set in categories.items():
            p1_counts[category] = self._count_word_occurrences(p1_text, word_set)
            p2_counts[category] = self._count_word_occurrences(p2_text, word_set)
        
        p1_counts["conjunctions"] = self._count_conjunctions(p1_text)
        p2_counts["conjunctions"] = self._count_conjunctions(p2_text)
        
        scores = {}
        final_val = 0
        
        for category in list(categories.keys()) + ["conjunctions"]:
            p1_rate = p1_counts[category] / p1_word_count
            p2_rate = p2_counts[category] / p2_word_count
            val = 1 - (abs(p1_rate - p2_rate) / (p1_rate + p2_rate + 0.0001))
            scores[category] = val
            final_val += val
        
        final_val /= len(scores)
        
        return {
            "avg_lsm_score": final_val,
            "category_scores": scores,
            "p1_counts": p1_counts,
            "p2_counts": p2_counts
        }