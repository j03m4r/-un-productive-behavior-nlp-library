"""
Persona Validation — Analysis 2: Between-Persona Differentiation

Question: Are the 7 personas distinguishable from each other?

Methods:
    1. Correlation-based feature filtering (removes redundant features before classification)
    2. Per-feature ANOVA — statistical significance of persona differences
    3. Random Forest classifier with leave-one-seed-out cross-validation
       Classification accuracy is the differentiation metric (chance = 1/7 ≈ 14.3%)
    4. Confusion matrix analysis — check whether confusion pattern follows
       the Walton dialogue-type decision tree
    5. Pairwise effect sizes (Cohen's d) for targeted persona-pair claims

Input: consistency_results.json (with data_array per persona)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway

from statsmodels.stats.outliers_influence import variance_inflation_factor

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


def _get_personas(results: dict) -> list[str]:
    return [p for p in PERSONA_ORDER if p in results]


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0: Correlation-Based Feature Filtering
# ═══════════════════════════════════════════════════════════════════════════════

def filter_correlated_features(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = 5.0,  # VIF threshold; 5-10 is conventional
) -> tuple[np.ndarray, list[str], list[tuple[str, float]]]:
    """
    Iterative removal of features with high Variance Inflation Factor (VIF).
    At each step, computes VIF for all remaining features and drops the one
    with the highest VIF above the threshold. Repeats until all features
    are below the threshold.
    Args:
        X: (n_samples, n_features) data matrix
        feature_names: list of feature names matching X columns
        threshold: VIF above which a feature is dropped (conventional: 5 or 10)
    Returns:
        X_filtered: (n_samples, n_kept_features)
        kept_names: list of surviving feature names
        dropped: list of (dropped_feature, vif_value) tuples
    """
    indices = list(range(X.shape[1]))
    dropped = []

    while True:
        X_sub = X[:, indices]
        vifs = [variance_inflation_factor(X_sub, i) for i in range(X_sub.shape[1])]
        max_vif = max(vifs)
        if max_vif < threshold:
            break
        worst = vifs.index(max_vif)
        dropped.append((feature_names[indices[worst]], float(max_vif)))
        indices.pop(worst)

    kept_names = [feature_names[i] for i in indices]
    X_filtered = X[:, indices]

    print(f"\n  VIF filter (threshold: {threshold}):")
    print(f"    {len(kept_names)}/{X.shape[1]} features retained")
    if dropped:
        print(f"    Dropped {len(dropped)} features:")
        for feat, vif in dropped:
            print(f"      {feat:42s} VIF={vif:.2f}")

    return X_filtered, kept_names, dropped

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Build Classification Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def build_classification_dataset(
    results: dict,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X, y, seed_ids arrays for classification.

    Returns:
        X: (700, n_features)
        y: (700,) persona labels
        seed_ids: (700,) seed indices for leave-one-seed-out
    """
    personas = _get_personas(results)

    X_parts = []
    y_parts = []
    seed_parts = []

    for persona in personas:
        data_array = np.array(results[persona]["data_array"])
        conv_means = data_array.mean(axis=1)

        n_conv = conv_means.shape[0]
        X_parts.append(conv_means)
        y_parts.append(np.array([persona] * n_conv))
        seed_parts.append(np.arange(n_conv))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    seed_ids = np.concatenate(seed_parts)

    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(personas)} personas, {len(np.unique(seed_ids))} seeds")

    return X, y, seed_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Per-Feature ANOVA
# ═══════════════════════════════════════════════════════════════════════════════

def run_anova(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Per-feature one-way ANOVA across personas."""
    personas = sorted(np.unique(y))
    n_features = X.shape[1]

    anova_results = {}
    for f_idx, f_name in enumerate(feature_names):
        groups = [X[y == p, f_idx] for p in personas]
        try:
            f_stat, p_value = f_oneway(*groups)
            if np.isnan(f_stat):
                f_stat, p_value = 0.0, 1.0
        except Exception:
            f_stat, p_value = 0.0, 1.0
        anova_results[f_name] = {"f_stat": float(f_stat), "p_value": float(p_value)}

    n_significant = sum(1 for v in anova_results.values() if v["p_value"] < 0.001)

    print(f"\n  ANOVA Results:")
    print(f"    {n_significant}/{n_features} features significant at p < 0.001")

    sorted_anova = sorted(anova_results.items(), key=lambda x: x[1]["f_stat"], reverse=True)
    print(f"\n    Top 10 by F-statistic:")
    for f_name, vals in sorted_anova[:10]:
        print(f"      {f_name:42s} F={vals['f_stat']:>8.1f}  p={vals['p_value']:.2e}")

    return {
        "per_feature_anova": anova_results,
        "n_significant": n_significant,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Random Forest with Leave-One-Seed-Out CV
# ═══════════════════════════════════════════════════════════════════════════════

def run_classifier(
    X: np.ndarray,
    y: np.ndarray,
    seed_ids: np.ndarray,
    feature_names: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Random Forest with grouped k-fold CV (seed-grouped 80/20 split).

    Seeds are shuffled and divided into n_folds equal groups. Each fold
    holds out one group of seeds and trains on the rest. With 100 seeds
    and 5 folds, each fold holds out 20 seeds (140 conversations) and
    trains on 80 seeds (560 conversations).

    This prevents information leakage from shared seed content while
    providing a more realistic train/test ratio than leave-one-out.
    """
    unique_seeds = sorted(np.unique(seed_ids))
    n_seeds = len(unique_seeds)
    personas = sorted(np.unique(y))

    rng = np.random.RandomState(random_state)
    shuffled_seeds = rng.permutation(unique_seeds)
    seed_folds = np.array_split(shuffled_seeds, n_folds)

    all_preds = []
    all_true = []

    seeds_per_fold = [len(f) for f in seed_folds]
    print(f"\n  Running {n_folds}-fold grouped CV "
          f"(seeds per fold: {seeds_per_fold}, "
          f"train/test: {n_seeds - seeds_per_fold[0]}/{seeds_per_fold[0]})...")

    for fold_idx, held_out_seeds in enumerate(seed_folds):
        test_mask = np.isin(seed_ids, held_out_seeds)
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)

        fold_acc = accuracy_score(y_test, preds)
        print(f"    Fold {fold_idx + 1}/{n_folds}: "
              f"train={train_mask.sum()}, test={test_mask.sum()}, "
              f"accuracy={fold_acc:.3f}")

        all_preds.extend(preds)
        all_true.extend(y_test)

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    accuracy = accuracy_score(all_true, all_preds)
    f1_macro = f1_score(all_true, all_preds, average="macro")
    f1_per_class = {}
    for p in personas:
        f1_per_class[p] = float(f1_score(all_true == p, all_preds == p, average="binary"))

    cm = confusion_matrix(all_true, all_preds, labels=personas)

    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    clf_full = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    clf_full.fit(X_scaled_full, y)
    importances = clf_full.feature_importances_

    print(f"\n  Classifier Results:")
    print(f"    Overall accuracy: {accuracy:.3f} (chance = {1/len(personas):.3f})")
    print(f"    Macro F1: {f1_macro:.3f}")
    print(f"\n    Per-persona F1:")
    for p in personas:
        label = p.replace("_", " ").title()
        print(f"      {label:25s} F1 = {f1_per_class.get(p, 0):.3f}")

    print(f"\n    Top 15 features by importance:")
    sorted_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for f_name, imp in sorted_imp[:15]:
        print(f"      {f_name:42s} importance = {imp:.4f}")

    print(f"\n    Classification report:")
    report = classification_report(
        all_true, all_preds,
        labels=personas,
        target_names=[p.replace("_", " ").title() for p in personas],
    )
    print(report)

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm.tolist(),
        "feature_importances": dict(zip(feature_names, importances.tolist())),
        "personas": personas,
        "all_predictions": all_preds.tolist(),
        "all_true_labels": all_true.tolist(),
    }


# Pairwise Effect Sizes (Cohen's d)
def compute_pairwise_effect_sizes(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict:
    personas = sorted(np.unique(y))
    effect_sizes = {}

    for i, p_a in enumerate(personas):
        for j, p_b in enumerate(personas):
            if j <= i:
                continue

            X_a = X[y == p_a]
            X_b = X[y == p_b]
            n_a, n_b = X_a.shape[0], X_b.shape[0]

            pair_effects = {}
            for f_idx, f_name in enumerate(feature_names):
                mean_a = X_a[:, f_idx].mean()
                mean_b = X_b[:, f_idx].mean()
                sd_a = X_a[:, f_idx].std(ddof=1)
                sd_b = X_b[:, f_idx].std(ddof=1)

                pooled_sd = np.sqrt(
                    ((n_a - 1) * sd_a**2 + (n_b - 1) * sd_b**2) / (n_a + n_b - 2)
                )

                d = (mean_a - mean_b) / pooled_sd if pooled_sd > 1e-10 else 0.0
                pair_effects[f_name] = float(d)

            effect_sizes[(p_a, p_b)] = pair_effects

    print(f"\n  Pairwise Effect Sizes (top per pair):")
    for (p_a, p_b), effects in sorted(effect_sizes.items()):
        sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)
        top_feat, top_d = sorted_effects[0]
        label_a = p_a.replace("_", " ").title()
        label_b = p_b.replace("_", " ").title()
        size = "large" if abs(top_d) > 0.8 else "medium" if abs(top_d) > 0.5 else "small"
        print(f"    {label_a:20s} vs {label_b:20s}: "
              f"d={top_d:+.2f} on {top_feat} ({size})")

    return effect_sizes


def plot_confusion_matrix(classifier_results, output_dir="./output/plots"):
    output_dir = _ensure_dir(output_dir)
    personas = classifier_results["personas"]
    cm = np.array(classifier_results["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # norm = LogNorm(vmin=0.001, vmax=1)

    n = len(personas)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap=plt.cm.Blues, aspect="equal", vmin=0, vmax=100)

    labels = [p.replace("_", " ").title() for p in personas]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)

    for i, persona in enumerate(personas):
        ax.get_yticklabels()[i].set_color(PERSONA_COLORS.get(persona, "black"))
        ax.get_xticklabels()[i].set_color(PERSONA_COLORS.get(persona, "black"))

    for i in range(n):
        for j in range(n):
            raw = cm[i, j]
            pct = cm_norm[i, j]
            text_color = "white" if pct > 50 else "black"
            ax.text(j, i, f"{raw}", ha="center", va="center",
                    fontsize=9, color=text_color,
                    fontweight="bold" if i == j else "normal")

    ax.set_xlabel("Predicted Persona", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Persona", fontsize=12, fontweight="bold")

    accuracy = classifier_results["accuracy"]
    f1 = classifier_results["f1_macro"]
    ax.set_title(
        f"Persona Classification — 5-Fold Grouped CV (80/20 Seed Split)\n"
        f"Accuracy: {accuracy:.1%} | Macro F1: {f1:.3f}",
        fontsize=13, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label="# Predictions")
    plt.tight_layout()
    path = output_dir / "7_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

FEATURE_CATEGORIES = {
    # Structural
    "word_count":          "Structural",
    "sentence_count":      "Structural",
    "avg_word_len":        "Structural",
    "question_count":      "Structural",
    "exclamation_count":   "Structural",
    "question_ratio":      "Structural",
    # Pronoun ratios
    "first_person_ratio":      "Pronoun",
    "second_person_ratio":     "Pronoun",
    "inclusive_pronoun_ratio": "Pronoun",
    # Argumentative structure
    "discourse_connectives_count": "Argumentative",
    "stance_adverbials_count":     "Argumentative",
    "reasoning_verbs_count":       "Argumentative",
    "modal_verbs_count":           "Argumentative",
    "full_root_clauses_count":     "Argumentative",
    "partial_root_clauses_count":  "Argumentative",
    "negation_count":              "Argumentative",
    # Dialogue acts
    "concession_count":    "Dialogue Acts",
    "challenge_count":     "Dialogue Acts",
    "proposal_count":      "Dialogue Acts",
    "acknowledgment_count":"Dialogue Acts",
    # Sentiment
    "sentiment_neg":      "Sentiment",
    "sentiment_neu":      "Sentiment",
    "sentiment_pos":      "Sentiment",
    "sentiment_compound": "Sentiment",
    # Toxicity
    "toxicity_score": "Toxicity",
}

def _get_category(feature_name: str) -> str:
    if feature_name.startswith("politeness_"):
        return "Politeness"
    return FEATURE_CATEGORIES.get(feature_name, "Other")


CATEGORY_COLORS = {
    "Structural":    "#4e79a7",
    "Pronoun":       "#f28e2b",
    "Argumentative": "#59a14f",
    "Dialogue Acts": "#e15759",
    "Politeness":    "#b07aa1",
    "Sentiment":     "#76b7b2",
    "Toxicity":      "#ff9da7",
    "Other":         "#bab0ac",
}

def plot_feature_importances(classifier_results, output_dir="./output/plots", top_k=20):
    output_dir = _ensure_dir(output_dir)
    importances = classifier_results["feature_importances"]
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_k]
    sorted_imp.reverse()  # horizontal bar: lowest importance at bottom

    raw_names = [f for f, _ in sorted_imp]
    values    = [v for _, v in sorted_imp]
    categories = [_get_category(f) for f in raw_names]
    bar_colors = [CATEGORY_COLORS[c] for c in categories]

    display_names = [
        f.replace("politeness_", "p:").replace("_", " ")
        for f in raw_names
    ]

    fig, ax = plt.subplots(figsize=(12, max(6, top_k * 0.38)))

    bars = ax.barh(
        range(len(display_names)), values,
        color=bar_colors, edgecolor="white", height=0.7,
    )
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title(
        f"Top {top_k} Features for Persona Classification (Random Forest)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    x_max = max(values) if values else 1.0
    for bar, val, cat in zip(bars, values, categories):
        bar_right = bar.get_x() + bar.get_width()
        bar_mid_y = bar.get_y() + bar.get_height() / 2

        ax.text(
            bar_right + x_max * 0.01, bar_mid_y,
            f"{val:.3f}",
            ha="left", va="center", fontsize=8,
        )
        pill_x    = bar.get_x() + x_max * 0.005
        text_color = "white"
        ax.text(
            pill_x, bar_mid_y,
            f"[{cat}]",
            ha="left", va="center", fontsize=7,
            color=text_color, style="italic",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=CATEGORY_COLORS[cat],
                edgecolor="none",
                alpha=0.85,
            ),
        )

    seen = {}
    for cat, col in CATEGORY_COLORS.items():
        if cat in set(categories):
            seen[cat] = col

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=col, label=cat)
        for cat, col in seen.items()
    ]
    ax.legend(
        handles=legend_handles,
        title="Category",
        loc="lower right",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout()
    path = output_dir / "8_feature_importances.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_pairwise_effect_size_matrix(effect_sizes, feature_name, output_dir="./output/plots"):
    output_dir = _ensure_dir(output_dir)

    all_personas = set()
    for p_a, p_b in effect_sizes.keys():
        all_personas.add(p_a)
        all_personas.add(p_b)
    personas = [p for p in PERSONA_ORDER if p in all_personas]
    n = len(personas)

    matrix = np.zeros((n, n))
    for i, p_a in enumerate(personas):
        for j, p_b in enumerate(personas):
            if i == j:
                continue
            if (p_a, p_b) in effect_sizes:
                matrix[i, j] = effect_sizes[(p_a, p_b)].get(feature_name, 0)
            elif (p_b, p_a) in effect_sizes:
                matrix[i, j] = -effect_sizes[(p_b, p_a)].get(feature_name, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 1.0)
    im = ax.imshow(matrix, cmap=plt.cm.RdBu, vmin=-vmax, vmax=vmax, aspect="equal")

    labels = [p.replace("_", " ").title() for p in personas]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(n):
        for j in range(n):
            if i != j:
                d = matrix[i, j]
                color = "white" if abs(d) > vmax * 0.6 else "black"
                ax.text(j, i, f"{d:+.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    display_name = feature_name.replace("politeness_", "p:").replace("_", " ").title()
    ax.set_title(
        f"Pairwise Cohen's d — {display_name}\n"
        f"(positive = row persona higher than column persona)",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label="Cohen's d")
    plt.tight_layout()

    safe_name = feature_name.replace(".", "_").replace(" ", "_")
    path = output_dir / f"9_effect_size_{safe_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def run_analysis_2(
    results: dict,
    feature_names: list[str],
    output_dir: str = "./output/plots",
    corr_threshold: float = 5.0,
    n_folds: int = 5,
    effect_size_plot_features: list[str] = None,
) -> dict:
    output_dir = _ensure_dir(output_dir)

    print("=" * 70)
    print("ANALYSIS 2: Between-Persona Differentiation")
    print("=" * 70)

    # Step 0: Build dataset
    print("\nStep 0: Building classification dataset...")
    X, y, seed_ids = build_classification_dataset(results, feature_names)

    # Step 1: Correlation filter
    print("\nStep 1: Filtering correlated features...")
    X_filtered, filtered_names, dropped = filter_correlated_features(
        X, feature_names, threshold=corr_threshold
    )

    # Step 2: ANOVA on filtered features
    print("\nStep 2: Per-feature ANOVA (on filtered features)...")
    anova_results = run_anova(X_filtered, y, filtered_names)

    # Step 3: Classifier on filtered features
    print("\nStep 3: Random Forest Classifier (grouped k-fold CV)...")
    classifier_results = run_classifier(X_filtered, y, seed_ids, filtered_names, n_folds=n_folds)

    # Step 4: Effect sizes on ALL original features (not filtered)
    print("\nStep 4: Pairwise Effect Sizes (all original features)...")
    effect_sizes = compute_pairwise_effect_sizes(X, y, feature_names)

    # Step 5: Visualizations
    print("\nStep 5: Generating visualizations...")
    plot_confusion_matrix(classifier_results, output_dir)
    plot_feature_importances(classifier_results, output_dir)

    if effect_size_plot_features is None:
        sorted_imp = sorted(
            classifier_results["feature_importances"].items(),
            key=lambda x: x[1], reverse=True,
        )
        effect_size_plot_features = [f for f, _ in sorted_imp[:3]]

    for f_name in effect_size_plot_features:
        plot_pairwise_effect_size_matrix(effect_sizes, f_name, output_dir)

    # Save
    effect_sizes_serializable = {
        f"{p_a}_vs_{p_b}": effects
        for (p_a, p_b), effects in effect_sizes.items()
    }

    save_results = {
        "feature_filtering": {
            "threshold": corr_threshold,
            "original_features": len(feature_names),
            "retained_features": len(filtered_names),
            "retained_feature_names": filtered_names,
            "dropped": [
                {"feature": f, "vif": vif}
                for f, vif in dropped
            ],
        },
        "cv_config": {
            "method": "grouped_k_fold",
            "n_folds": n_folds,
            "seeds_per_fold": 100 // n_folds,
            "train_test_ratio": f"{100 - 100 // n_folds}/{100 // n_folds}",
        },
        "anova": {
            "n_significant_features": anova_results["n_significant"],
            "per_feature": anova_results["per_feature_anova"],
        },
        "classifier": {
            "accuracy": classifier_results["accuracy"],
            "f1_macro": classifier_results["f1_macro"],
            "f1_per_class": classifier_results["f1_per_class"],
            "confusion_matrix": classifier_results["confusion_matrix"],
            "feature_importances": classifier_results["feature_importances"],
            "personas": classifier_results["personas"],
        },
        "effect_sizes": effect_sizes_serializable,
    }

    results_path = output_dir / "analysis_2_results.json"
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return save_results

if __name__ == "__main__":
    RESULTS_PATH = "./modules/persona_validation/output/persona_consistency/consistency_results.json"
    OUTPUT_DIR = "./modules/persona_validation/output/persona_differentiation/plots"

    feature_names = [
        "word_count", "sentence_count", "avg_word_len",
        "question_count", "exclamation_count", "question_ratio",
        "first_person_ratio", "second_person_ratio", "inclusive_pronoun_ratio",
        "discourse_connectives_count", "stance_adverbials_count",
        "reasoning_verbs_count", "modal_verbs_count",
        "full_root_clauses_count", "partial_root_clauses_count",
        "negation_count",
        "concession_count", "challenge_count", "proposal_count", "acknowledgment_count",
        "politeness_Hedges", "politeness_Impersonal.Pronoun", "politeness_Swearing",
        "politeness_Negation", "politeness_Filler.Pause", "politeness_Informal.Title",
        "politeness_Formal.Title", "politeness_Could.You", "politeness_Can.You",
        "politeness_By.The.Way", "politeness_Let.Me.Know", "politeness_Goodbye",
        "politeness_For.Me", "politeness_For.You", "politeness_Reasoning",
        "politeness_Contrast.Conjunction", "politeness_Reassurance",
        "politeness_Ask.Agency", "politeness_Give.Agency", "politeness_Hello",
        "politeness_Please", "politeness_First.Person.Plural",
        "politeness_First.Person.Single", "politeness_Second.Person",
        "politeness_Third.Person", "politeness_Positive.Emotion",
        "politeness_Negative.Emotion", "politeness_Agreement",
        "politeness_Disagreement", "politeness_Acknowledgement",
        "politeness_Subjectivity", "politeness_Bare.Command",
        "politeness_WH.Questions", "politeness_Repair.Questions",
        "politeness_Tag.Questions", "politeness_YesNo.Questions",
        "politeness_Gratitude", "politeness_Apology",
        "politeness_Truth.Intensifier", "politeness_Adverb.Limiter",
        "politeness_Affirmation", "politeness_Conjunction.Start",
        "sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment_compound",
        "toxicity_score",
    ]

    effect_size_features = [
        "question_ratio",
        "proposal_count",
        "toxicity_score",
        "concession_count",
    ]

    print(f"Loading results from {RESULTS_PATH}...")
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    run_analysis_2(
        results, feature_names,
        output_dir=OUTPUT_DIR,
        corr_threshold=10.0,
        n_folds=5,
        effect_size_plot_features=effect_size_features,
    )