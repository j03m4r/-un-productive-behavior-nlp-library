from modules import Evaluator
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score

def cosine_similarity_lexical(text1, text2):
    """Computes how similar the two texts are based on what words they use"""
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return float(similarity)

def semantic_similarity(text1, text2):
    """Computes how similar the meaning of both texts are (even with different words)"""
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding1 = sentence_model.encode(text1)
    embedding2 = sentence_model.encode(text2)
    
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return float(similarity)

def bertscore_similarity(text1, text2):
    P, R, F1 = bert_score([text1], [text2], lang="en", verbose=False)
    
    return {
        "precision": P.item(),
        "recall": R.item(),
        "f1": F1.item()
    }

class RelevanceEvaluator(Evaluator):
    def __init__(self):
        super().__init__(name="Relevance")

    def evaluate_utterance_pair(self, text1: str, text2: str) -> dict:
        cosine_sim = cosine_similarity_lexical(text1, text2)
        semantic_sim =semantic_similarity(text1, text2)
        # bert_scores = bertscore_similarity(text1, text2)
        
        return {
            "cosine_similarity": cosine_sim,
            "semantic_similarity": semantic_sim,
            # "bertscore": bert_scores
        }

    def evaluate_conversation(self, conversation):
        """Counts how many utterances are irrelevant (cosine and semantic similarity 
        values <= 0.25) to the first utterance in the conversation"""
        num_irrelevant_utterances = 0
        text1 = conversation[0]
        for text in conversation[1:len(conversation)]:
            res = self.evaluate_utterance_pair(text1, text)
            if (res["cosine_similarity"] <= 0.25 and res["semantic_similarity"] <= 0.25):
                num_irrelevant_utterances += 1
        return num_irrelevant_utterances