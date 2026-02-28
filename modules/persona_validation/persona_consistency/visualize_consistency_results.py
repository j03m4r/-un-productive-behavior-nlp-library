"""
Persona Validation — Consistency Visualizations

Four visualizations for Analysis 1 (Within-Persona Consistency):
    1. Persona feature profile bar chart
    2. Trajectory consistency heatmap (personas x features)
    3. Per-turn trajectory plots (7 personas per feature)
    4. Stratification comparison (overall vs within-stratum)

Usage:
    from consistency_visualizations import plot_all
    plot_all(results, feature_names, output_dir="./output/plots")

    Or run standalone:
    python consistency_visualizations.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy.stats import f_oneway

# ── Consistent styling ──────────────────────────────────────────────────────

PERSONA_COLORS = {
    "persuasion":          "#e7298a",
    "eristic":             "#d95f02",
    "deliberation":        "#1b9e77",
    "inquiry":             "#7570b3",
    "negotiation":         "#a6761d",
    "discovery":           "#66a61e",
    "information_seeking": "#e6ab02",
}

PERSONA_ORDER = [
    "persuasion", "negotiation", "deliberation", "inquiry",
    "eristic", "discovery", "information_seeking",
]

# Features grouped by category for readable visualizations
FEATURE_GROUPS = {
    "Structural": [
        "word_count", "sentence_count", "avg_word_len",
        "question_count", "exclamation_count", "question_ratio",
    ],
    "Pronouns": [
        "first_person_ratio", "second_person_ratio", "inclusive_pronoun_ratio",
    ],
    "Argumentative": [
        "discourse_connectives_count", "stance_adverbials_count",
        "reasoning_verbs_count", "modal_verbs_count",
        "full_root_clauses_count", "partial_root_clauses_count",
        "negation_count",
    ],
    "Dialogue Acts": [
        "concession_count", "challenge_count",
        "proposal_count", "acknowledgment_count",
    ],
    "Politeness Strategies": [
            "politeness_Hedges",
            "politeness_Impersonal.Pronoun",
            "politeness_Swearing",
            "politeness_Negation",
            "politeness_Filler.Pause",
            "politeness_Informal.Title",
            "politeness_Formal.Title",
            "politeness_Could.You",
            "politeness_Can.You",
            "politeness_By.The.Way",
            "politeness_Let.Me.Know",
            "politeness_Goodbye",
            "politeness_For.Me",
            "politeness_For.You",
            "politeness_Reasoning",
            "politeness_Contrast.Conjunction",
            "politeness_Reassurance",
            "politeness_Ask.Agency",
            "politeness_Give.Agency",
            "politeness_Hello",
            "politeness_Please",
            "politeness_First.Person.Plural",
            "politeness_First.Person.Single",
            "politeness_Second.Person",
            "politeness_Third.Person",
            "politeness_Positive.Emotion",
            "politeness_Negative.Emotion",
            "politeness_Agreement",
            "politeness_Disagreement",
            "politeness_Acknowledgement",
            "politeness_Subjectivity",
            "politeness_Bare.Command",
            "politeness_WH.Questions",
            "politeness_Repair.Questions",
            "politeness_Tag.Questions",
            "politeness_YesNo.Questions",
            "politeness_Gratitude",
            "politeness_Apology",
            "politeness_Truth.Intensifier",
            "politeness_Adverb.Limiter",
            "politeness_Affirmation",
            "politeness_Conjunction.Start",
    ],
    "Sentiment": [
        "sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment_compound",
    ],
    "Toxicity": [
        "toxicity_score",
    ],
}

KEY_FEATURES = None

feature_names = [
    "word_count",
    "sentence_count",
    "avg_word_len",
    "question_count",
    "exclamation_count",
    "question_ratio",
    "first_person_ratio",
    "second_person_ratio",
    "inclusive_pronoun_ratio",
    "discourse_connectives_count",
    "stance_adverbials_count",
    "reasoning_verbs_count",
    "modal_verbs_count",
    "full_root_clauses_count",
    "partial_root_clauses_count",
    "negation_count",
    "concession_count",
    "challenge_count",
    "proposal_count",
    "acknowledgment_count",
    "politeness_Hedges",
    "politeness_Impersonal.Pronoun",
    "politeness_Swearing",
    "politeness_Negation",
    "politeness_Filler.Pause",
    "politeness_Informal.Title",
    "politeness_Formal.Title",
    "politeness_Could.You",
    "politeness_Can.You",
    "politeness_By.The.Way",
    "politeness_Let.Me.Know",
    "politeness_Goodbye",
    "politeness_For.Me",
    "politeness_For.You",
    "politeness_Reasoning",
    "politeness_Contrast.Conjunction",
    "politeness_Reassurance",
    "politeness_Ask.Agency",
    "politeness_Give.Agency",
    "politeness_Hello",
    "politeness_Please",
    "politeness_First.Person.Plural",
    "politeness_First.Person.Single",
    "politeness_Second.Person",
    "politeness_Third.Person",
    "politeness_Positive.Emotion",
    "politeness_Negative.Emotion",
    "politeness_Agreement",
    "politeness_Disagreement",
    "politeness_Acknowledgement",
    "politeness_Subjectivity",
    "politeness_Bare.Command",
    "politeness_WH.Questions",
    "politeness_Repair.Questions",
    "politeness_Tag.Questions",
    "politeness_YesNo.Questions",
    "politeness_Gratitude",
    "politeness_Apology",
    "politeness_Truth.Intensifier",
    "politeness_Adverb.Limiter",
    "politeness_Affirmation",
    "politeness_Conjunction.Start",
    "sentiment_neg",
    "sentiment_neu",
    "sentiment_pos",
    "sentiment_compound",
    "toxicity_score"
]

def _get_personas(results: dict) -> list[str]:
    """Return persona names in consistent order."""
    return [p for p in PERSONA_ORDER if p in results]


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_anova_f_statistics(
    results: dict,
    feature_names: list[str],
) -> dict[str, dict]:
    """
    For each feature, compute one-way ANOVA F-statistic across personas.

    For each persona, we compute per-conversation means (mean over 4 turns),
    giving 100 values per persona. Then we test whether the 7 groups differ.

    Returns:
        dict mapping feature_name -> {"f_stat": float, "p_value": float}
    """
    personas = _get_personas(results)

    # Build per-persona conversation-level means for each feature
    # For each persona: mean over 4 turns -> shape (n_conversations, n_features)
    persona_conv_means = {}
    for persona in personas:
        mean_matrix = np.array(results[persona]["per_turn_stability"]["mean"])  # (4, n_features)
        sd_matrix = np.array(results[persona]["per_turn_stability"]["sd"])      # (4, n_features)

        # We need per-conversation means, not the already-aggregated mean over conversations.
        # The per_turn_stability mean is already averaged over conversations,
        # so we can't recover individual conversation values from it.
        # Instead, we need the raw data_array. Check if it's available in results,
        # otherwise fall back to using the aggregated mean as a proxy.
        #
        # If raw data is available (stored during wrapper execution):
        if "data_array" in results[persona]:
            data_array = np.array(results[persona]["data_array"])  # (n_conv, 4, n_features)
            conv_means = data_array.mean(axis=1)  # (n_conv, n_features)
        else:
            # Fallback: we only have aggregated stats.
            # Simulate per-conversation means using the stored mean and sd.
            # This is a rough approximation — for proper ANOVA you should
            # store the data_array in your wrapper results.
            n_conv = results[persona]["n_conversations"]
            grand_mean = mean_matrix.mean(axis=0)  # (n_features,)
            pooled_sd = sd_matrix.mean(axis=0)      # (n_features,)
            rng = np.random.default_rng(seed=hash(persona) % 2**31)
            conv_means = rng.normal(
                loc=grand_mean, scale=pooled_sd / np.sqrt(4), size=(n_conv, len(feature_names))
            )

        persona_conv_means[persona] = conv_means

    # Run one-way ANOVA per feature
    f_results = {}
    for f_idx, f_name in enumerate(feature_names):
        groups = [
            persona_conv_means[persona][:, f_idx]
            for persona in personas
        ]

        # Check if all groups are constant (zero variance) — ANOVA undefined
        all_constant = all(np.std(g) < 1e-10 for g in groups)
        if all_constant:
            f_results[f_name] = {"f_stat": 0.0, "p_value": 1.0}
            continue

        try:
            f_stat, p_value = f_oneway(*groups)
            # Handle NaN from degenerate cases
            if np.isnan(f_stat):
                f_stat = 0.0
                p_value = 1.0
        except Exception:
            f_stat = 0.0
            p_value = 1.0

        f_results[f_name] = {"f_stat": float(f_stat), "p_value": float(p_value)}

    return f_results


def compute_consistency_spread(
    results: dict,
    feature_names: list[str],
) -> dict[str, dict]:
    """
    For each feature, compute the standard deviation of trajectory consistency
    (mean cosine) across the 7 personas.

    Features where some personas are highly consistent and others aren't
    will have high spread — these reveal persona-specific behavioral patterns.

    Returns:
        dict mapping feature_name -> {
            "spread": float (SD across personas),
            "per_persona": dict[str, float] (mean cosine per persona)
        }
    """
    personas = _get_personas(results)

    spread_results = {}
    for f_name in feature_names:
        per_persona = {}
        for persona in personas:
            tc = results[persona]["trajectory_consistency"]["per_feature"]
            val = tc.get(f_name, {}).get("mean_cosine", np.nan)
            per_persona[persona] = val

        values = [v for v in per_persona.values() if not np.isnan(v)]

        if len(values) < 2:
            spread = 0.0
        else:
            spread = float(np.std(values, ddof=1))

        spread_results[f_name] = {
            "spread": spread,
            "per_persona": per_persona,
        }

    return spread_results

def identify_key_features(
    results: dict,
    feature_names: list[str],
    top_k: int = 10,
    anova_p_threshold: float = 0.001,
    activity_threshold: float = 0.05,
) -> list[dict]:
    """
    Identify the most influential features for persona differentiation.

    Strategy:
        1. Filter out near-zero features (max grand mean < activity_threshold)
        2. Filter out features that don't significantly differentiate personas
           (ANOVA p-value > anova_p_threshold)
        3. Among survivors, rank by average of ANOVA rank and consistency
           spread rank
        4. Take top_k from the combined ranking

    This ensures every selected feature both differentiates personas AND
    has meaningfully different consistency patterns across personas.

    Args:
        results: Output from calculate_consistency_metrics (with data_array)
        feature_names: List of all feature names
        top_k: Number of top features to select
        anova_p_threshold: Only features with p < this survive
        activity_threshold: Only features with max grand mean > this survive

    Returns:
        List of dicts sorted by combined rank, each with:
            - "feature": feature name
            - "f_stat": ANOVA F-statistic
            - "f_p_value": ANOVA p-value
            - "consistency_spread": SD of trajectory consistency across personas
            - "anova_rank": rank by F-stat among survivors (1 = highest)
            - "spread_rank": rank by spread among survivors (1 = highest)
            - "combined_rank": average of anova_rank and spread_rank
    """
    print("\nIdentifying key features...")

    personas = _get_personas(results)

    # ── Step 1: Activity filter ─────────────────────────────────────────────
    # Remove near-zero features that produce spurious statistics
    all_grand = []
    for persona in personas:
        data_array = np.array(results[persona]["data_array"])
        all_grand.append(data_array.mean(axis=(0, 1)))
    all_grand = np.array(all_grand)

    max_per_feature = all_grand.max(axis=0)
    active_features = set()
    for i, f_name in enumerate(feature_names):
        if max_per_feature[i] > activity_threshold:
            active_features.add(f_name)

    n_total = len(feature_names)
    n_active = len(active_features)
    print(f"  Activity filter: {n_active}/{n_total} features survive "
          f"(threshold: max grand mean > {activity_threshold})")

    # ── Step 2: Compute ANOVA and filter by significance ────────────────────
    anova_results = compute_anova_f_statistics(results, feature_names)
    spread_results = compute_consistency_spread(results, feature_names)

    significant_features = set()
    for f_name, vals in anova_results.items():
        if f_name in active_features and vals["p_value"] < anova_p_threshold:
            significant_features.add(f_name)

    n_significant = len(significant_features)
    print(f"  ANOVA filter: {n_significant}/{n_active} active features are significant "
          f"(p < {anova_p_threshold})")

    if n_significant == 0:
        print("  WARNING: No features survived both filters.")
        return []

    # ── Step 3: Rank survivors by both criteria ─────────────────────────────
    # Rank by ANOVA F-stat (descending) among survivors only
    survivor_anova = sorted(
        [(f, anova_results[f]["f_stat"]) for f in significant_features],
        key=lambda x: x[1],
        reverse=True,
    )
    anova_rank_map = {f: rank + 1 for rank, (f, _) in enumerate(survivor_anova)}

    # Rank by consistency spread (descending) among survivors only
    survivor_spread = sorted(
        [(f, spread_results[f]["spread"]) for f in significant_features],
        key=lambda x: x[1],
        reverse=True,
    )
    spread_rank_map = {f: rank + 1 for rank, (f, _) in enumerate(survivor_spread)}

    # ── Step 4: Combined ranking, take top_k ────────────────────────────────
    candidates = []
    for f_name in significant_features:
        a_rank = anova_rank_map[f_name]
        s_rank = spread_rank_map[f_name]
        combined = (a_rank + s_rank) / 2

        feature_group = None
        for group_name, group_features in FEATURE_GROUPS.items():
            if f_name in group_features:
                feature_group = group_name
                break

        candidates.append({
            "feature": f_name,
            "feature_group": feature_group,
            "f_stat": anova_results[f_name]["f_stat"],
            "f_p_value": anova_results[f_name]["p_value"],
            "consistency_spread": spread_results[f_name]["spread"],
            "anova_rank": a_rank,
            "spread_rank": s_rank,
            "combined_rank": combined,
        })

    candidates.sort(key=lambda x: x["combined_rank"])
    selected = candidates[:top_k]

    # ── Print summary ───────────────────────────────────────────────────────
    print(f"\n  Top {top_k} key features (from {n_significant} survivors):")
    print(f"  {'#':<4s} {'Feature':<42s} {'F-stat':>8s} {'p-value':>10s} "
          f"{'Spread':>8s} {'ANOVA#':>7s} {'Spread#':>8s} {'Combined':>9s}")
    print(f"  {'─'*4} {'─'*42} {'─'*8} {'─'*10} {'─'*8} {'─'*7} {'─'*8} {'─'*9}")

    for i, kf in enumerate(selected):
        p_str = f"{kf['f_p_value']:.2e}" if kf["f_p_value"] < 0.001 else f"{kf['f_p_value']:.4f}"
        print(
            f"  {i+1:<4d} {kf['feature']:<42s} "
            f"{kf['f_stat']:>8.1f} {p_str:>10s} "
            f"{kf['consistency_spread']:>8.3f} "
            f"{kf['anova_rank']:>7d} {kf['spread_rank']:>8d} "
            f"{kf['combined_rank']:>9.1f}"
        )

    # Also show what got filtered out that might be surprising
    filtered_by_activity = set(feature_names) - active_features
    filtered_by_anova = active_features - significant_features
    if filtered_by_anova:
        notable_filtered = sorted(
            [(f, anova_results[f]["f_stat"], anova_results[f]["p_value"])
             for f in filtered_by_anova],
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        print(f"\n  Notable features filtered by ANOVA (top 5 by F-stat among rejects):")
        for f, fstat, pval in notable_filtered:
            print(f"    {f:<42s} F={fstat:.1f}  p={pval:.4f}")

    return selected


def deduplicate_features(
    results: dict,
    key_features: list[dict],
    feature_names: list[str],
    corr_threshold: float = 0.9,
) -> list[dict]:
    """
    Remove redundant features from the key features list based on
    correlation of their grand-mean profiles across personas.

    When two features correlate above the threshold, the one with the
    worse combined_rank is dropped.

    Args:
        results: consistency results with data_array
        key_features: output from identify_key_features, sorted by combined_rank
        feature_names: full list of feature names
        corr_threshold: features with |r| above this are considered redundant

    Returns:
        Filtered list of key features (same format as input, fewer items)
    """
    personas = _get_personas(results)

    # Build grand mean matrix: (n_personas, n_features)
    all_grand = []
    for persona in personas:
        data_array = np.array(results[persona]["data_array"])
        all_grand.append(data_array.mean(axis=(0, 1)))
    all_grand = np.array(all_grand)

    # Compute correlation matrix
    # Handle constant features (zero std) that produce NaN correlations
    corr_matrix = np.corrcoef(all_grand.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Get indices for key features
    kf_names = [kf["feature"] for kf in key_features]
    kf_indices = [feature_names.index(f) for f in kf_names]

    # Greedy deduplication: iterate in rank order, drop later features
    # that correlate too highly with already-kept features
    kept = []
    dropped = []

    for kf in key_features:
        f_idx = feature_names.index(kf["feature"])

        # Check correlation with all already-kept features
        is_redundant = False
        redundant_with = None

        for kept_kf in kept:
            kept_idx = feature_names.index(kept_kf["feature"])
            r = abs(corr_matrix[f_idx, kept_idx])

            if r > corr_threshold and kf["feature_group"] == kept_kf.get("feature_group"):
                is_redundant = True
                redundant_with = kept_kf["feature"]
                break

        if is_redundant:
            dropped.append((kf["feature"], redundant_with, r))
        else:
            kept.append(kf)

    # Report
    if dropped:
        print(f"\n  Deduplication (|r| > {corr_threshold}):")
        print(f"  Kept {len(kept)}, dropped {len(dropped)}:")
        for feat, redundant_with, r in dropped:
            print(f"    Dropped {feat:40s} (r={r:.3f} with {redundant_with})")
    else:
        print(f"\n  Deduplication: no redundant features found (threshold: {corr_threshold})")

    return kept


def _order_features_by_group(features: list[str]) -> tuple[list[str], list[int], list[str]]:
    """
    Reorder features according to FEATURE_GROUPS, with ungrouped features at the end.
    Splits ungrouped into politeness vs other.

    Returns:
        ordered: reordered feature list
        boundaries: indices where each group starts
        labels: group names
    """
    ordered = []
    boundaries = []
    labels = []

    for group_name, group_features in FEATURE_GROUPS.items():
        valid = [f for f in group_features if f in features]
        if valid:
            boundaries.append(len(ordered))
            labels.append(group_name)
            ordered.extend(valid)

    remaining = [f for f in features if f not in ordered]
    non_politeness = [f for f in remaining if not f.startswith("politeness_")]
    politeness = [f for f in remaining if f.startswith("politeness_")]

    if non_politeness:
        boundaries.append(len(ordered))
        labels.append("Other")
        ordered.extend(non_politeness)

    if politeness:
        boundaries.append(len(ordered))
        labels.append("Politeness")
        ordered.extend(politeness)

    return ordered, boundaries, labels


def _plot_single_heatmap(
    results: dict,
    personas: list[str],
    features: list[str],
    title: str,
    filename: str,
    output_dir: Path,
    annotate: bool = True,
    figsize_per_feature: float = 0.6,
    fontsize_annotations: int = 9,
    fontsize_xlabels: int = 8,
    group_features: bool = True,
):
    """Shared heatmap rendering logic."""

    if not features:
        print(f"  Skipping {filename}: no features to plot")
        return

    n_personas = len(personas)

    if group_features:
        ordered, boundaries, group_labels = _order_features_by_group(features)
    else:
        ordered = features
        boundaries = []
        group_labels = []

    n_features = len(ordered)

    # Build matrix
    matrix = np.full((n_personas, n_features), np.nan)
    for p_idx, persona in enumerate(personas):
        tc = results[persona]["trajectory_consistency"]["per_feature"]
        for f_idx, f_name in enumerate(ordered):
            if f_name in tc:
                matrix[p_idx, f_idx] = tc[f_name]["mean_cosine"]

    # Figure sizing
    fig_width = max(8, n_features * figsize_per_feature + 2)
    fig_height = max(4, n_personas * 0.7 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#f0f0f0")

    im = ax.imshow(matrix, cmap=cmap, vmin=-0.2, vmax=1.0, aspect="auto")

    # Y labels (personas)
    ax.set_yticks(range(n_personas))
    ax.set_yticklabels(
        [p.replace("_", " ").title() for p in personas],
        fontsize=11, fontweight="bold",
    )

    # Color persona labels
    for p_idx, persona in enumerate(personas):
        ax.get_yticklabels()[p_idx].set_color(PERSONA_COLORS[persona])

    # X labels (features)
    ax.set_xticks(range(n_features))
    display_names = []
    for f in ordered:
        name = f.replace("politeness_", "p:").replace("sentiment_", "s:")
        # For key features heatmap, use cleaner names
        name = name.replace("_count", "").replace("_ratio", " ratio").replace("_", " ")
        display_names.append(name)

    ax.set_xticklabels(
        display_names,
        fontsize=fontsize_xlabels,
        rotation=45,
        ha="right",
    )

    # Annotate cells
    if annotate:
        for i in range(n_personas):
            for j in range(n_features):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > 0.8 or val < 0.2 else "black"
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=fontsize_annotations,
                        color=text_color,
                        fontweight="bold",
                    )

    # Group separators and labels
    if boundaries:
        for boundary in boundaries[1:]:
            ax.axvline(x=boundary - 0.5, color="black", linewidth=1.5)

        for g_idx, (boundary, label) in enumerate(zip(boundaries, group_labels)):
            next_boundary = (
                boundaries[g_idx + 1] if g_idx + 1 < len(boundaries) else n_features
            )
            mid = (boundary + next_boundary - 1) / 2
            ax.text(
                mid, -0.9, label,
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#444444",
            )

    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean Cosine Similarity", pad=0.02)

    ax.set_title(
        f"{title}\n(higher = more consistent 4-turn pattern across conversations)",
        fontsize=13, fontweight="bold", pad=30 if boundaries else 15,
    )

    plt.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    
def plot_split_heatmaps(
    results: dict,
    feature_names: list[str],
    key_feature_names: list[str],
    output_dir: str = "./output/plots",
):
    """
    Two heatmaps:
        A) Key features only — large cells, fully annotated, paper-ready
        B) All remaining features — compact reference, appendix material

    Both show trajectory consistency (mean cosine similarity) per persona × feature.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    personas = _get_personas(results)
    n_personas = len(personas)

    # Split features
    remaining_features = [f for f in feature_names if f not in key_feature_names]

    # ── Heatmap A: Key Features ─────────────────────────────────────────────
    _plot_single_heatmap(
        results=results,
        personas=personas,
        features=key_feature_names,
        title="Trajectory Consistency — Key Differentiating Features",
        filename="2a_heatmap_key_features.png",
        output_dir=output_dir,
        annotate=True,
        figsize_per_feature=0.8,
        fontsize_annotations=10,
        fontsize_xlabels=9,
        group_features=True,
    )

    # ── Heatmap B: Remaining Features ───────────────────────────────────────
    # Group remaining features for readability
    _plot_single_heatmap(
        results=results,
        personas=personas,
        features=remaining_features,
        title="Trajectory Consistency — All Other Features",
        filename="2b_heatmap_remaining_features.png",
        output_dir=output_dir,
        annotate=False,
        figsize_per_feature=0.35,
        fontsize_annotations=6,
        fontsize_xlabels=7,
        group_features=True,
    )



# ── Visualization 3: Per-Turn Trajectory Plots ──────────────────────────────

def plot_per_turn_trajectories(
    results: dict,
    feature_names: list[str],
    output_dir: str = "./modules/persona_validation/output/persona_consistency/plots",
):
    """
    For each key feature, plot the 4-turn trajectory for all 7 personas
    on the same axes (mean ± 1 SD band).

    Shows how personas diverge or converge over the course of a conversation.
    """
    output_dir = _ensure_dir(output_dir)
    personas = _get_personas(results)
    features_to_plot = [f for f in KEY_FEATURES if f in feature_names]

    # Layout: grid of subplots
    n_plots = len(features_to_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    turns = np.arange(1, 5)

    for plot_idx, f_name in enumerate(features_to_plot):
        ax = axes[plot_idx]
        f_idx = feature_names.index(f_name)

        for persona in personas:
            mean_matrix = np.array(results[persona]["per_turn_stability"]["mean"])
            sd_matrix = np.array(results[persona]["per_turn_stability"]["sd"])

            means = mean_matrix[:, f_idx]
            sds = sd_matrix[:, f_idx]

            color = PERSONA_COLORS[persona]
            label = persona.replace("_", " ").title()

            ax.plot(turns, means, marker="o", markersize=4,
                    color=color, label=label, linewidth=2)
            ax.fill_between(turns, means - sds, means + sds,
                            color=color, alpha=0.1)

        ax.set_xlabel("Turn", fontsize=9)
        ax.set_ylabel(f_name.replace("_", " ").title(), fontsize=9)
        ax.set_title(f_name.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xticks(turns)
        ax.set_xticklabels(["T1", "T2", "T3", "T4"])
        ax.grid(alpha=0.3)

    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        ncol=len(personas), fontsize=8,
        bbox_to_anchor=(0.5, 1.02),
    )

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Per-Turn Feature Trajectories by Persona (Mean ± 1 SD)",
        fontsize=14, fontweight="bold", y=1.05,
    )
    plt.tight_layout()
    path = output_dir / "3_per_turn_trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Visualization 4: Stratification Comparison ──────────────────────────────

def plot_stratification_comparison(
    results: dict,
    feature_names: list[str],
    output_dir: str = "./modules/persona_validation/output/persona_consistency/plots",
):
    """
    For each persona, compare overall trajectory consistency vs
    within-stratum consistency for key features.

    Shows whether apparent inconsistency is really appropriate adaptation
    to seed type.
    """
    output_dir = _ensure_dir(output_dir)
    personas = _get_personas(results)
    features_to_plot = [f for f in KEY_FEATURES if f in feature_names]

    strata_names = ["wiki_attack", "wiki_no_attack", "cmv_attack", "cmv_no_attack"]
    strata_colors = {
        "wiki_attack":    "#d62828",
        "wiki_no_attack": "#f77f00",
        "cmv_attack":     "#003049",
        "cmv_no_attack":  "#219ebc",
    }

    n_personas = len(personas)
    n_cols = 2
    n_rows = int(np.ceil(n_personas / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    for p_idx, persona in enumerate(personas):
        ax = axes[p_idx]

        # Overall consistency per feature
        overall_tc = results[persona]["trajectory_consistency"]["per_feature"]
        overall_vals = []
        for f in features_to_plot:
            val = overall_tc.get(f, {}).get("mean_cosine", np.nan)
            overall_vals.append(val)

        x = np.arange(len(features_to_plot))
        bar_width = 0.15

        # Plot overall as wider background bar
        ax.bar(
            x, overall_vals, width=0.7,
            color="lightgray", edgecolor="gray", linewidth=0.5,
            label="Overall", zorder=1,
        )

        # Plot each stratum as narrow bars on top
        stratified = results[persona].get("stratified", {})
        for s_idx, stratum in enumerate(strata_names):
            if stratum not in stratified:
                continue
            s_tc = stratified[stratum]["trajectory_consistency"]["per_feature"]
            s_vals = []
            for f in features_to_plot:
                val = s_tc.get(f, {}).get("mean_cosine", np.nan)
                s_vals.append(val)

            offset = (s_idx - len(strata_names) / 2 + 0.5) * bar_width
            ax.bar(
                x + offset, s_vals, width=bar_width,
                color=strata_colors[stratum], edgecolor="white", linewidth=0.3,
                label=stratum.replace("_", " ").title(), zorder=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f.replace("_", "\n") for f in features_to_plot],
            fontsize=7, rotation=45, ha="right",
        )
        ax.set_ylabel("Mean Cosine Similarity", fontsize=9)
        ax.set_title(
            persona.replace("_", " ").title(),
            fontsize=12, fontweight="bold",
            color=PERSONA_COLORS[persona],
        )
        ax.set_ylim(-0.3, 1.1)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
        ax.grid(axis="y", alpha=0.3)

        if p_idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    # Hide unused subplots
    for idx in range(n_personas, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Trajectory Consistency: Overall vs. Stratified by Seed Type\n"
        "(gray = overall, colored = within-stratum; higher stratum bars = consistent adaptation)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = output_dir / "4_stratification_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def plot_persona_profile_pca_3d(
    results: dict,
    feature_names: list[str],
    output_dir: str = "./modules/persona_validation/output/persona_consistency/plots",
    use_key_features: bool = True,
):
    """
    3D PCA visualization of persona feature profiles.

    Each persona is represented by its grand-mean feature vector
    (mean over turns and conversations), exactly matching the data
    used in the persona profile bar chart.

    PCA is performed across personas (N=7), not conversations.

    Args:
        results: consistency analysis output
        feature_names: full ordered feature list
        output_dir: directory to save plot
        use_key_features: if True, restrict PCA to KEY_FEATURES
    """
    output_dir = _ensure_dir(output_dir)
    personas = _get_personas(results)

    # Decide which features to include
    if use_key_features:
        if KEY_FEATURES is None:
            raise ValueError("KEY_FEATURES is None but use_key_features=True")
        features = [f for f in KEY_FEATURES if f in feature_names]
    else:
        features = feature_names

    if len(features) < 3:
        raise ValueError("Need at least 3 features for 3D PCA")

    # ── Build persona × feature matrix ──────────────────────────────────────
    X = []
    for persona in personas:
        mean_matrix = np.array(results[persona]["per_turn_stability"]["mean"])
        grand_mean = mean_matrix.mean(axis=0)  # (n_features,)

        persona_vector = [
            grand_mean[feature_names.index(f)]
            for f in features
        ]
        X.append(persona_vector)

    X = np.array(X)  # shape: (n_personas, n_features)

    # ── Standardize & PCA ───────────────────────────────────────────────────
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(9.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    for i, persona in enumerate(personas):
        ax.scatter(
            X_pca[i, 0],
            X_pca[i, 1],
            X_pca[i, 2],
            s=120,
            color=PERSONA_COLORS[persona],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
        )

        ax.text(
            X_pca[i, 0],
            X_pca[i, 1],
            X_pca[i, 2],
            persona.replace("_", " ").title(),
            fontsize=9,
            ha="center",
            va="center",
        )

    total_var = explained.sum() * 100

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=10)
    ax.set_zlabel(f"PC3 ({explained[2]*100:.1f}%)", fontsize=10, labelpad=8)

    ax.set_title(
        f"3D PCA of Persona Feature Profiles\n"
        f"(All Features, Grand Means — {total_var:.1f}% Variance Explained)",
        fontsize=14,
        fontweight="bold",
        pad=22,
    )

    # View + spacing fixes
    ax.view_init(elev=22, azim=-55)
    plt.subplots_adjust(right=0.88)

    plt.tight_layout()

    path = output_dir / "5_persona_profile_pca_3d.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")

"""
Persona Characterization & Z-Scored Visualization

Two functions:
    1. characterize_personas() — identifies top characteristic features and trajectory
       shifts per persona, prints a readable summary, returns structured data
    2. plot_persona_profiles_zscore() — z-scored grouped bar chart replacing the
       raw-value version that had the scale problem

Both filter out near-zero features that produce spurious z-scores.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PERSONA_ORDER = [
    "persuasion", "negotiation", "deliberation", "inquiry",
    "eristic", "discovery", "information_seeking",
]

ACTIVITY_THRESHOLD = 0.05  # features with max grand mean below this are filtered out

def _build_feature_matrices(results: dict):
    """
    Extract grand means and trajectory means from results, returning:
        feature_names: list of all feature names
        all_grand: (n_personas, n_features) grand means
        trajectory_means: dict[persona -> (4, n_features)]
    """
    personas = _get_personas(results)
    feature_names = list(results[personas[0]]["trajectory_consistency"]["per_feature"].keys())

    all_grand = []
    trajectory_means = {}
    for persona in personas:
        data_array = np.array(results[persona]["data_array"])
        all_grand.append(data_array.mean(axis=(0, 1)))
        trajectory_means[persona] = data_array.mean(axis=0)

    return feature_names, np.array(all_grand), trajectory_means


def _filter_active_features(feature_names, all_grand):
    """
    Remove features where the max value across all personas is below
    ACTIVITY_THRESHOLD. These produce spurious z-scores.

    Returns:
        active_names: filtered feature name list
        active_indices: indices into original feature_names
        active_grand: filtered (n_personas, n_active) array
    """
    max_per_feature = all_grand.max(axis=0)
    active_mask = max_per_feature > ACTIVITY_THRESHOLD
    active_indices = np.where(active_mask)[0]
    active_names = [feature_names[i] for i in active_indices]
    active_grand = all_grand[:, active_mask]
    return active_names, active_indices, active_grand


def _compute_zscores(active_grand):
    """Z-score each feature across personas (axis=0)."""
    feat_mean = active_grand.mean(axis=0)
    feat_std = active_grand.std(axis=0, ddof=0)
    feat_std[feat_std < 1e-10] = 1.0
    z_scored = (active_grand - feat_mean) / feat_std
    return z_scored, feat_mean, feat_std


# ── Characterize Personas ───────────────────────────────────────────────────

def characterize_personas(
    results: dict,
    top_k_static: int = 3,
    top_k_shift: int = 3,
) -> dict:
    """
    For each persona, identify:
        1. Top characteristic features (highest |z-score|) — what makes this
           persona different from the average persona
        2. Top trajectory shifts (biggest T1→T4 change) — how this persona's
           behavior evolves over the course of a conversation

    Args:
        results: Output from calculate_consistency_metrics (with data_array)
        top_k_static: Number of top characteristic features per persona
        top_k_shift: Number of top trajectory shifts per persona

    Returns:
        dict mapping persona -> {
            "characteristics": [{"feature", "z_score", "direction", "raw_value", "global_mean"}, ...],
            "trajectory_shifts": [{"feature", "normalized_shift", "direction", "t1", "t4"}, ...],
        }
    """
    personas = _get_personas(results)
    feature_names, all_grand, trajectory_means = _build_feature_matrices(results)
    active_names, active_indices, active_grand = _filter_active_features(feature_names, all_grand)
    z_scored, feat_mean, feat_std = _compute_zscores(active_grand)

    print("=" * 90)
    print("PERSONA CHARACTERIZATION SUMMARY")
    print(f"(Active features: {len(active_names)}/{len(feature_names)}, "
          f"activity threshold: {ACTIVITY_THRESHOLD})")
    print("=" * 90)

    persona_chars = {}

    for p_idx, persona in enumerate(personas):
        z = z_scored[p_idx]

        # ── Static characteristics ──
        ranked_static = sorted(range(len(active_names)), key=lambda i: abs(z[i]), reverse=True)
        characteristics = []
        for r in ranked_static[:top_k_static]:
            characteristics.append({
                "feature": active_names[r],
                "z_score": float(z[r]),
                "direction": "HIGH" if z[r] > 0 else "LOW",
                "raw_value": float(active_grand[p_idx, r]),
                "global_mean": float(feat_mean[r]),
            })

        # ── Trajectory shifts ──
        tm = trajectory_means[persona][:, active_indices]  # (4, n_active)
        raw_shift = tm[3] - tm[0]
        normalized_shift = raw_shift / feat_std

        ranked_shift = sorted(range(len(active_names)), key=lambda i: abs(normalized_shift[i]), reverse=True)
        shifts = []
        for r in ranked_shift[:top_k_shift]:
            shifts.append({
                "feature": active_names[r],
                "normalized_shift": float(normalized_shift[r]),
                "direction": "↑" if normalized_shift[r] > 0 else "↓",
                "t1": float(tm[0, r]),
                "t4": float(tm[3, r]),
            })

        persona_chars[persona] = {
            "characteristics": characteristics,
            "trajectory_shifts": shifts,
        }

        # ── Print ──
        label = persona.replace("_", " ").upper()
        print(f"\n  {label}")
        print(f"  {'─' * 86}")
        print(f"  Characteristic features:")
        for c in characteristics:
            print(
                f"    {c['feature']:42s}  z={c['z_score']:+.2f} ({c['direction']})  "
                f"raw={c['raw_value']:.3f}  avg={c['global_mean']:.3f}"
            )
        print(f"  Trajectory shifts (T1 → T4):")
        for s in shifts:
            print(
                f"    {s['feature']:42s}  shift={s['normalized_shift']:+.2f} ({s['direction']})  "
                f"{s['t1']:.2f} → {s['t4']:.2f}"
            )

    return persona_chars


# ── Z-Scored Feature Profile Bar Chart ──────────────────────────────────────

def plot_persona_profiles_zscore(
    results: dict,
    key_feature_names: list[str] = None,
    output_dir: str = "./output/plots",
):
    """
    Grouped bar chart of z-scored persona feature profiles.

    Z-scoring normalizes all features to the same scale (mean=0, sd=1 across
    personas), so word_count and question_ratio are visually comparable.

    Bars above 0 = persona is higher than average on this feature.
    Bars below 0 = persona is lower than average.

    Args:
        results: consistency results with data_array
        key_feature_names: if provided, only plot these features.
            If None, uses all active features (may be crowded).
        output_dir: where to save the plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    personas = _get_personas(results)
    feature_names, all_grand, _ = _build_feature_matrices(results)
    active_names, active_indices, active_grand = _filter_active_features(feature_names, all_grand)
    z_scored, _, _ = _compute_zscores(active_grand)

    # Select features to plot
    if key_feature_names is not None:
        plot_features = [f for f in key_feature_names if f in active_names]
    else:
        plot_features = active_names

    if not plot_features:
        print("Warning: no features to plot after filtering")
        return

    # Get indices into active_names
    plot_indices = [active_names.index(f) for f in plot_features]

    n_features = len(plot_features)
    n_personas = len(personas)

    fig, ax = plt.subplots(figsize=(max(14, n_features * 0.9), 7))

    x = np.arange(n_features)
    bar_width = 0.8 / n_personas

    for p_idx, persona in enumerate(personas):
        values = [z_scored[p_idx, i] for i in plot_indices]
        offset = (p_idx - n_personas / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, values, bar_width,
            label=persona.replace("_", " ").title(),
            color=PERSONA_COLORS[persona],
            edgecolor="white", linewidth=0.5,
        )

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f.replace("politeness_", "p:").replace("_", " ") for f in plot_features],
        fontsize=8, rotation=45, ha="right",
    )
    ax.set_ylabel("Z-Score (across personas)", fontsize=11)
    ax.set_title(
        "Persona Feature Profiles (Z-Scored)\n"
        "Bars above 0 = higher than average across personas",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(-3, 3)

    plt.tight_layout()
    path = output_dir / "1_persona_feature_profiles_zscore.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Persona Signature Cards ─────────────────────────────────────────────────

def plot_persona_signature_cards(
    results: dict,
    output_dir: str = "./output/plots",
    top_k: int = 5,
):
    """
    One subplot per persona showing its top characteristic features
    as horizontal bar charts of z-scores. Compact visual summary of
    what makes each persona distinctive.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    personas = _get_personas(results)
    feature_names, all_grand, _ = _build_feature_matrices(results)
    active_names, _, active_grand = _filter_active_features(feature_names, all_grand)
    z_scored, _, _ = _compute_zscores(active_grand)

    n_personas = len(personas)
    n_cols = 2
    n_rows = int(np.ceil(n_personas / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    for p_idx, persona in enumerate(personas):
        ax = axes[p_idx]
        z = z_scored[p_idx]

        # Top features by |z|
        ranked = sorted(range(len(active_names)), key=lambda i: abs(z[i]), reverse=True)
        top_indices = ranked[:top_k]
        top_indices.reverse()  # so highest is at top of horizontal bar chart

        labels = [
            active_names[i].replace("politeness_", "p:").replace("_", " ")
            for i in top_indices
        ]
        values = [z[i] for i in top_indices]
        colors = [
            "#2a9d8f" if v > 0 else "#e63946" for v in values
        ]

        bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white", height=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlim(-3, 3)
        ax.set_xlabel("Z-Score", fontsize=9)
        ax.set_title(
            persona.replace("_", " ").title(),
            fontsize=12, fontweight="bold",
            color=PERSONA_COLORS[persona],
        )
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        # Annotate values
        for bar, val in zip(bars, values):
            x_pos = val + (0.08 if val > 0 else -0.08)
            ha = "left" if val > 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.2f}", ha=ha, va="center", fontsize=8, fontweight="bold")

    # Hide unused
    for idx in range(n_personas, len(axes)):
        axes[idx].set_visible(False)

    # Legend
    high_patch = mpatches.Patch(color="#2a9d8f", label="Higher than average")
    low_patch = mpatches.Patch(color="#e63946", label="Lower than average")
    fig.legend(handles=[high_patch, low_patch], loc="upper center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        "Persona Signature Features (Top 5 by |Z-Score|)",
        fontsize=14, fontweight="bold", y=1.05,
    )
    plt.tight_layout()
    path = output_dir / "6_persona_signature_cards.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Standalone ──────────────────────────────────────────────────────────────
    # 1. Print characterization summary

    # 2. Z-scored bar chart (using key features or all active)
    # You can pass key_feature_names from identify_key_features() here

# ── Master function ─────────────────────────────────────────────────────────

def plot_all(
    results: dict,
    feature_names: list[str],
    output_dir: str = "./modules/persona_validation/output/persona_consistency/plots",
):
    """Generate all four consistency visualizations."""
    print("\nGenerating visualizations...")
    plot_split_heatmaps(results, feature_names, KEY_FEATURES, output_dir)
    plot_per_turn_trajectories(results, feature_names, output_dir)
    plot_stratification_comparison(results, feature_names, output_dir)
    plot_persona_profile_pca_3d(results, feature_names, output_dir, use_key_features=False)
    plot_persona_profiles_zscore(results, key_feature_names=KEY_FEATURES, output_dir=output_dir)
    plot_persona_signature_cards(results, output_dir=output_dir, top_k=5)
    print(f"\nAll visualizations saved to {output_dir}/")


# ── Standalone entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    RESULTS_PATH = "./modules/persona_validation/output/persona_consistency/consistency_results.json"

    print(f"Loading results from {RESULTS_PATH}...")
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    persona_chars = characterize_personas(results, top_k_static=5, top_k_shift=3)
    
    candidates = identify_key_features(results, feature_names, top_k=30)
    deduped = deduplicate_features(results, candidates, feature_names, 0.9)

    # Step 3: Take final_k
    final = deduped[:9]

    print(f"\n  Final {len(final)} key features after deduplication:")
    for i, kf in enumerate(final):
        print(f"    {i+1}. {kf['feature']}")

    KEY_FEATURES = [kf["feature"] for kf in final]
    KEY_FEATURES.remove("question_count")
    KEY_FEATURES.append("toxicity_score")
    KEY_FEATURES.remove("politeness_Please")
    KEY_FEATURES.append("word_count")

    plot_all(results, feature_names)