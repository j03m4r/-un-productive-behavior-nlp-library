from modules import Evaluator
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

# --- Module-level initialization ---
# Loaded once at import time. Do not re-instantiate inside functions.
_sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


def fit_lexical_vectorizer(texts: list[str]) -> CountVectorizer:
    """Fit a single CountVectorizer on all texts so the vector space is
    consistent across pairwise comparisons. Call this once on your full
    set of responses before computing any lexical similarities.

    Args:
        texts: All texts that will be compared (e.g. all 7 persona responses).

    Returns:
        A fitted CountVectorizer.
    """
    vectorizer = CountVectorizer()
    vectorizer.fit(texts)
    return vectorizer


def cosine_similarity_lexical(text1: str, text2: str, vectorizer: CountVectorizer) -> float:
    """Computes how similar two texts are based on word overlap,
    using a pre-fit vectorizer for a consistent vocabulary space.

    Args:
        text1: First text.
        text2: Second text.
        vectorizer: A CountVectorizer already fit on the full text set.

    Returns:
        Cosine similarity score in [0, 1].
    """
    vectors = vectorizer.transform([text1, text2])
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


def semantic_similarity(text1: str, text2: str) -> float:
    """Computes how similar the meaning of both texts are,
    even if they use different words. Uses the module-level
    SentenceTransformer (all-MiniLM-L6-v2).

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Cosine similarity of sentence embeddings in [-1, 1],
        typically [0, 1] for natural language.
    """
    embedding1 = _sentence_model.encode(text1)
    embedding2 = _sentence_model.encode(text2)
    return float(cosine_similarity([embedding1], [embedding2])[0][0])


def bertscore_similarity(text1: str, text2: str) -> dict:
    """Computes BERTScore between two texts. Note: significantly
    slower than the other two metrics due to contextual token alignment.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Dict with precision, recall, and F1 scores.
    """
    P, R, F1 = bert_score([text1], [text2], lang="en", verbose=False)
    return {
        "precision": P.item(),
        "recall": R.item(),
        "f1": F1.item()
    }


class RelevanceEvaluator(Evaluator):
    def __init__(self, texts: list[str] = None):
        """
        Args:
            texts: The full set of texts that will be compared. Used to fit
                   the lexical vectorizer once, so the vocabulary space is
                   consistent across all pairwise comparisons. If None, the
                   vectorizer will be fit lazily on the first conversation
                   passed to evaluate_conversation.
        """
        super().__init__(name="Relevance")
        self._vectorizer = fit_lexical_vectorizer(texts) if texts else None

    def _ensure_vectorizer(self, texts: list[str]):
        """Fit the vectorizer if it hasn't been initialized yet."""
        if self._vectorizer is None:
            self._vectorizer = fit_lexical_vectorizer(texts)

    def set_vectorizer(self, texts: list[str]):
        """Manually set (or reset) the lexical vectorizer by fitting
        it on the provided texts.

        Args:
            texts: All texts that will be compared.
        """
        self._vectorizer = fit_lexical_vectorizer(texts)

    def evaluate_utterance_pair(self, text1: str, text2: str) -> dict:
        """Compute lexical and semantic similarity between two texts.

        NOTE: If you call this standalone (outside of evaluate_conversation),
        you need to have passed texts to __init__, or the vectorizer won't
        be fit. If you're comparing ad-hoc pairs, consider passing all
        candidate texts at init time.

        Args:
            text1: First utterance.
            text2: Second utterance.

        Returns:
            Dict with lexical and sentence_embedding similarity scores.
        """
        if self._vectorizer is None:
            # Fallback: fit on just these two texts. This is inconsistent
            # if you're comparing multiple pairs — prefer passing all texts
            # to __init__ instead.
            self._vectorizer = fit_lexical_vectorizer([text1, text2])

        cosine_sim = cosine_similarity_lexical(text1, text2, self._vectorizer)
        semantic_sim = semantic_similarity(text1, text2)
        bert_scores = bertscore_similarity(text1, text2)

        return {
            "lexical": cosine_sim,
            "sentence_embeddings": semantic_sim,
            "bertscore": bert_scores
        }

    def evaluate_conversation(self, conversation: list[str]) -> int:
        """Counts how many utterances are irrelevant to the first utterance
        in the conversation.

        An utterance is considered irrelevant if BOTH cosine_similarity AND
        semantic_similarity are <= 0.25. This is a lenient criterion —
        an utterance only needs to score above 0.25 on one metric to be
        considered relevant. Adjust thresholds or switch to OR logic if
        you want stricter filtering.

        NOTE: This compares every utterance against conversation[0] only.
        If you want turn-to-turn relevance (each utterance vs. the one
        before it), that's a different computation.

        Args:
            conversation: List of utterance strings. conversation[0] is the
                          reference utterance.

        Returns:
            Number of utterances (excluding the first) deemed irrelevant.
        """
        self._ensure_vectorizer(conversation)

        num_irrelevant_utterances = 0
        text1 = conversation[0]

        for text in conversation[1:]:
            res = self.evaluate_utterance_pair(text1, text)
            if res["lexical"] <= 0.25 and res["sentence_embeddings"] <= 0.25:
                num_irrelevant_utterances += 1

        return num_irrelevant_utterances


# ---------------------------------------------------------------------------
# Main block: compute 7x7 similarity matrices and visualize
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DIALOGUE_TYPES = [
        "Persuasion",
        "Deliberation",
        "Inquiry",
        "Negotiation",
        "Eristic",
        "Discovery",
        "Information-Seeking",
    ]

    RESPONSES = {
        "Persuasion":          "I appreciate your passion, but calling me an idiot doesn't address the theological argument. The post makes a careful distinction—it's not condemning people for their orientation, only certain actions based on scriptural interpretation. How does discussing religious doctrine directly cause harm? Many faith traditions maintain ethical standards while still showing compassion. Can you explain the causal link between theological discussion and the violence you're referencing?",
        "Deliberation":        "I see strong disagreement here about whether the OP's theological distinction causes real-world harm to LGBTQ+ people. The responder sees the post as normalizing prejudice, while the OP might view it as expressing religious conviction without advocating discrimination. Is the core issue whether making this religious argument—even with the orientation/behavior distinction—contributes to harm against gay people?",
        "Inquiry":             "I hear you're really upset about this post. Before we dig into whether it actually normalizes harm, let's make sure I understand your position clearly: Are you saying this specific post will lead to horrific actions, or that posts like this contribute to a broader pattern that does? What's the mechanism you see connecting theological arguments to harmful behavior?",
        "Negotiation":         "I hear your concern about harm, but calling names doesn't help us understand each other. Here's what I'm thinking: maybe we can agree that theological beliefs about behavior shouldn't justify discrimination or violence against LGBTQ+ people? I maintain my view about biblical teachings on actions, but could we find common ground that people deserve safety and respect regardless of orientation? What matters most to you here?",
        "Eristic":             "So you think calling people \"idiots\" makes you the moral authority here? Your knee-jerk outrage completely sidesteps the actual theological argument being made. The post explicitly distinguished between orientation and behavior - but you're too busy virtue signaling to engage with nuance. If you can't handle complex discussions about religious doctrine without throwing tantrums, maybe you shouldn't participate in them",
        "Discovery":           "I'm curious what you think connects theological beliefs to real-world harm. When someone expresses a religious view distinguishing attraction from action, what specific mechanism do you see causing the \"horrific\" outcomes you're referencing? Are you suggesting the OP's framing itself enables violence, or that any public expression of this doctrine does, regardless of how it's stated?",
        "Information-Seeking": "I can see you feel strongly that this post causes real harm. What specifically do you think happens when people read messages like the OP's—how does it lead from someone's theological belief to actual harmful actions against gay people?",
    }

    # Ordered list so indexing is consistent across matrices
    labels = DIALOGUE_TYPES
    texts  = [RESPONSES[label] for label in labels]
    n = len(labels)

    # --- 1. Lexical cosine similarity matrix ---
    vectorizer = fit_lexical_vectorizer(texts)
    lexical_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                lexical_matrix[i][j] = 1.0
            elif j > i:
                score = cosine_similarity_lexical(texts[i], texts[j], vectorizer)
                lexical_matrix[i][j] = score
                lexical_matrix[j][i] = score  # symmetric

    # --- 2. Semantic similarity matrix (batch encode for speed) ---
    # encode() on a list is faster than calling it per-text individually.
    all_embeddings = _sentence_model.encode(texts)
    semantic_matrix = cosine_similarity(all_embeddings)  # returns full NxN matrix directly

    # --- 3. Visualization helpers ---
    COLORS = [
        "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
        "#f4a261", "#264653", "#6a4c93",
    ]

    def plot_heatmap(matrix: np.ndarray, title: str, ax: plt.Axes):
        """Render a labeled heatmap on the given axes."""
        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=1)

        # Tick labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)

        # Annotate cells with the score value
        for i in range(n):
            for j in range(n):
                val = matrix[i][j]
                text_color = "white" if val > 0.7 or val < 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=11, fontweight="bold")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        plt.colorbar(im, ax=ax, shrink=0.85, label="Similarity")

    def plot_pca_scatter_3d(matrix: np.ndarray, title: str, ax: Axes3D):
        """Project the similarity matrix into 3D via PCA on the row vectors,
        then plot a labeled 3D scatter. This captures more variance than 2D."""
        pca = PCA(n_components=3)
        coords = pca.fit_transform(matrix)

        for i, label in enumerate(labels):
            ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                       c=COLORS[i], s=150, edgecolors="black",
                       linewidths=1.5, depthshade=True)
            # 3D text annotation
            ax.text(coords[i, 0], coords[i, 1], coords[i, 2],
                    label, fontsize=9, fontweight="bold", color=COLORS[i])

        total_var = sum(pca.explained_variance_ratio_) * 100
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)", fontsize=10)
        ax.set_title(f"{title}\n(Total variance: {total_var:.1f}%)",
                     fontsize=12, fontweight="bold", pad=15)
        ax.grid(True, linestyle="--", alpha=0.3)

    # --- 4. Render: 2 rows x 2 cols (heatmap + 3D PCA per metric) ---
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Persona Similarity Across Walton's Dialogue Types",
                 fontsize=16, fontweight="bold", y=0.98)

    # Row 1: Lexical
    ax1 = plt.subplot(2, 2, 1)
    plot_heatmap(lexical_matrix, "Lexical Cosine Similarity", ax1)
    
    ax2 = plt.subplot(2, 2, 2, projection='3d')
    plot_pca_scatter_3d(lexical_matrix, "Lexical — 3D PCA", ax2)

    # Row 2: Semantic
    ax3 = plt.subplot(2, 2, 3)
    plot_heatmap(semantic_matrix, "Semantic Similarity (all-MiniLM-L6-v2)", ax3)
    
    ax4 = plt.subplot(2, 2, 4, projection='3d')
    plot_pca_scatter_3d(semantic_matrix, "Semantic — 3D PCA", ax4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("persona_similarity.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- 5. Terminal: ranked pairs ---
    print("\n" + "=" * 70)
    print(" PAIRWISE RANKINGS (most different → most similar)")
    print("=" * 70)
    for metric_name, matrix in [("Lexical", lexical_matrix),
                                 ("Semantic", semantic_matrix)]:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((labels[i], labels[j], matrix[i][j]))
        pairs.sort(key=lambda x: x[2])

        print(f"\n--- {metric_name} ---")
        for a, b, score in pairs:
            flag = "🔴" if score > 0.75 else "🟡" if score > 0.55 else "🟢"
            print(f"  {flag}  {a:25s} ↔ {b:25s}  {score:.3f}")