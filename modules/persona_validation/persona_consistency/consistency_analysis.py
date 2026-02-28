import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from collections import defaultdict

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

def generate_conversation_data_array(persona_conversations: list[dict], feature_names: list[str]) -> np.ndarray:
    """
    Convert list of conversation dicts into a 3D numpy array.

    Args:
        persona_conversations: list of conversation dicts with extracted features per turn.
        feature_names: list of feature name strings.

    Returns:
        np.ndarray of shape (n_conversations, 4, n_features)
    """
    n_conversations = len(persona_conversations)
    n_features = len(feature_names)
    data_array = np.zeros((n_conversations, 4, n_features))

    for i, convo in enumerate(persona_conversations):
        for j, turn in enumerate(convo["turns"]):
            features = turn["features"]
            missing = set(feature_names) - set(features.keys())
            assert not missing, (
                f"Conversation {convo['seed_id']} turn {j} missing features: {missing}"
            )
            data_array[i, j] = [features[name] for name in feature_names]

    return data_array

def calculate_per_turn_stability_across_conversations(data_array: np.ndarray):
    """
    Calculates per-turn stability metrics across conversations.

    Args:
        data_array: shape (n_conversations, n_turns_per_conversation, n_features)

    Returns:
        data_array: (n_conversations, n_turns_per_conversation, n_features)
        mean: (4, n_features)
        sd:   (4, n_features)
        cv:   (4, n_features)
        cv_per_feature: (n_features,)
    """

    # Metric A: per-turn-position stability across conversations
    mean = data_array.mean(axis=0)
    sd   = data_array.std(axis=0, ddof=1)

    eps = 1e-5
    cv = sd / (mean + eps)

    cv_per_feature = cv.mean(axis=0)

    return mean, sd, cv, cv_per_feature

def calculate_trajectory_consistency(data_array: np.ndarray, feature_names: list[str]) -> dict:
    """
    For each feature, compute how similar the 4-turn trajectories are
    across all conversations for a single persona.

    Args:
        data_array: shape (n_conversations, 4, n_features)
        feature_names: list of feature name strings, length n_features

    Returns:
        dict with:
            - "per_feature": dict mapping feature_name -> {
                "mean_cosine": float,
                "median_cosine": float,
                "std_cosine": float,
                "n_valid_pairs": int,
                "n_zero_trajectories": int
              }
            - "summary_mean": float (mean of all per-feature mean cosines)
    """
    n_conversations, n_turns, n_features = data_array.shape
    assert n_features == len(feature_names), (
        f"data_array has {n_features} features but feature_names has {len(feature_names)}"
    )

    per_feature = {}

    for f_idx, f_name in enumerate(feature_names):
        # Step 1: Extract all trajectories for this feature
        # trajectories shape: (n_conversations, 4)
        # Each row is one conversation's 4-turn trajectory for this feature
        trajectories = data_array[:, :, f_idx]

        # Step 2: Identify and remove zero vectors
        # Cosine similarity is undefined for zero vectors (division by zero).
        # A zero trajectory means this feature was 0 at all 4 turns — e.g.,
        # challenge_count = [0,0,0,0] for an information-seeking persona.
        # We track how many were removed for reporting.
        norms = np.linalg.norm(trajectories, axis=1)
        nonzero_mask = norms > 1e-10
        valid_trajectories = trajectories[nonzero_mask]
        n_zero = int((~nonzero_mask).sum())

        # Step 3: Handle edge cases
        # If fewer than 2 valid trajectories, we can't compute any pairwise similarity
        if len(valid_trajectories) < 2:
            per_feature[f_name] = {
                "mean_cosine": np.nan,
                "median_cosine": np.nan,
                "std_cosine": np.nan,
                "n_valid_pairs": 0,
                "n_zero_trajectories": n_zero,
            }
            continue

        # Step 4: Compute pairwise cosine similarity
        # Returns matrix of shape (n_valid, n_valid) where entry (i,j) is
        # the cosine similarity between conversation i's trajectory and
        # conversation j's trajectory.
        sim_matrix = cosine_similarity(valid_trajectories)

        # Step 5: Extract upper triangle (excluding diagonal)
        # The diagonal is always 1.0 (self-similarity), and the matrix is
        # symmetric, so we only need the upper triangle to avoid counting
        # each pair twice.
        upper_indices = np.triu_indices_from(sim_matrix, k=1)
        pairwise_sims = sim_matrix[upper_indices]

        # Step 6: Compute summary statistics over all pairwise similarities
        per_feature[f_name] = {
            "mean_cosine": float(np.mean(pairwise_sims)),
            "median_cosine": float(np.median(pairwise_sims)),
            "std_cosine": float(np.std(pairwise_sims, ddof=1)),
            "n_valid_pairs": len(pairwise_sims),
            "n_zero_trajectories": n_zero,
        }

    # Step 7: Overall summary — mean across features, skipping NaN
    valid_means = [
        v["mean_cosine"] for v in per_feature.values()
        if not np.isnan(v["mean_cosine"])
    ]
    summary_mean = float(np.mean(valid_means)) if valid_means else np.nan

    return {
        "per_feature": per_feature,
        "summary_mean": summary_mean,
    }

def stratify_conversations(conversations: list[dict]) -> dict[str, list[dict]]:
    """
    Split conversations into 4 strata based on platform and attack presence.

    Returns:
        dict mapping stratum name -> list of conversations
    """
    strata = defaultdict(list)
    for convo in conversations:
        platform = convo["platform"]
        has_attack = convo["has_attack"]
        key = f"{platform}_{'attack' if has_attack else 'no_attack'}"
        strata[key].append(convo)
    return dict(strata)

def calculate_consistency_metrics(
    conversations_by_persona: dict[str, list[dict]],
) -> dict:
    """
    Main function to calculate consistency metrics for a set of conversations.

    Args:
        conversations_by_persona: Dictionary mapping persona names to lists of
            conversation dicts with extracted features per turn.

    Returns:
        Dictionary of consistency metrics for each persona, including
        overall and stratified results.
    """
    results = {}

    for persona, conversations in conversations_by_persona.items():
        print(f"\n{'='*60}")
        print(f"Persona: {persona} ({len(conversations)} conversations)")
        print(f"{'='*60}")

        # --- Overall metrics ---
        data_array = generate_conversation_data_array(conversations, feature_names)
        mean, sd, cv, cv_per_feature = calculate_per_turn_stability_across_conversations(data_array)
        trajectory_consistency = calculate_trajectory_consistency(data_array, feature_names)

        # Print top 10 most/least consistent features by trajectory cosine
        print(f"\n  Top 10 most consistent features (trajectory cosine):")
        sorted_features = sorted(
            trajectory_consistency["per_feature"].items(),
            key=lambda x: x[1]["mean_cosine"] if not np.isnan(x[1]["mean_cosine"]) else -1,
            reverse=True,
        )
        for name, vals in sorted_features[:10]:
            print(f"    {name:45s} mean_cos={vals['mean_cosine']:.3f}  (zeros={vals['n_zero_trajectories']})")

        print(f"\n  Top 10 least consistent features (trajectory cosine):")
        # Filter out NaN before sorting ascending
        valid_sorted = [
            (name, vals) for name, vals in sorted_features if not np.isnan(vals["mean_cosine"])
        ]
        for name, vals in valid_sorted[-10:]:
            print(f"    {name:45s} mean_cos={vals['mean_cosine']:.3f}  (zeros={vals['n_zero_trajectories']})")

        # Print features with all-zero trajectories
        all_zero_features = [
            name for name, vals in trajectory_consistency["per_feature"].items()
            if vals["n_zero_trajectories"] > len(conversations) * 0.5
        ]
        if all_zero_features:
            print(f"\n  Features mostly zero (>50% zero trajectories): {all_zero_features}")

        # --- Stratified metrics ---
        strata = stratify_conversations(conversations)
        stratified_results = {}

        print(f"\n  Stratified consistency:")
        for stratum_name, stratum_convos in sorted(strata.items()):
            if len(stratum_convos) < 5:
                print(f"    {stratum_name}: skipped (only {len(stratum_convos)} conversations)")
                continue

            stratum_array = generate_conversation_data_array(stratum_convos, feature_names)
            stratum_mean, stratum_sd, stratum_cv, stratum_cv_per_feature = (
                calculate_per_turn_stability_across_conversations(stratum_array)
            )
            stratum_trajectory = calculate_trajectory_consistency(stratum_array, feature_names)

            stratified_results[stratum_name] = {
                "n_conversations": len(stratum_convos),
                "per_turn_stability": {
                    "mean": stratum_mean,
                    "sd": stratum_sd,
                    "cv": stratum_cv,
                    "cv_per_feature": stratum_cv_per_feature,
                },
                "trajectory_consistency": stratum_trajectory,
            }

            print(
                f"    {stratum_name:25s} n={len(stratum_convos):3d}  "
                f"trajectory_summary={stratum_trajectory['summary_mean']:.3f}"
            )

        # Compare stratified vs overall
        overall_summary = trajectory_consistency["summary_mean"]
        stratum_summaries = [
            v["trajectory_consistency"]["summary_mean"]
            for v in stratified_results.values()
            if not np.isnan(v["trajectory_consistency"]["summary_mean"])
        ]
        if stratum_summaries:
            mean_stratum_summary = np.mean(stratum_summaries)
            print(
                f"\n  Overall trajectory consistency:      {overall_summary:.3f}"
                f"\n  Mean within-stratum consistency:     {mean_stratum_summary:.3f}"
                f"\n  Difference (stratified - overall):   {mean_stratum_summary - overall_summary:+.3f}"
            )
            if mean_stratum_summary - overall_summary > 0.05:
                print("  → Personas are adapting consistently to seed type (good sign)")
            elif mean_stratum_summary - overall_summary < -0.01:
                print("  → Warning: stratification decreased consistency (unexpected)")
            else:
                print("  → Minimal stratification effect (persona is stable regardless of seed)")

        # Store results
        results[persona] = {
            "data_array": data_array.tolist(),
            "n_conversations": len(conversations),
            "per_turn_stability": {
                "mean": mean,
                "sd": sd,
                "cv": cv,
                "cv_per_feature": cv_per_feature,
            },
            "trajectory_consistency": trajectory_consistency,
            "stratified": stratified_results,
        }

    return results


def save_results(results: dict, output_dir: str = "./output"):
    """
    Save results to disk in both JSON (for programmatic use) and
    CSV (for quick inspection in a spreadsheet).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Save trajectory consistency as CSV ---
    # Rows: (persona, feature), Columns: mean_cosine, median_cosine, std_cosine, etc.
    csv_rows = []
    for persona, persona_results in results.items():
        for f_name, f_vals in persona_results["trajectory_consistency"]["per_feature"].items():
            csv_rows.append({
                "persona": persona,
                "feature": f_name,
                "mean_cosine": f_vals["mean_cosine"],
                "median_cosine": f_vals["median_cosine"],
                "std_cosine": f_vals["std_cosine"],
                "n_valid_pairs": f_vals["n_valid_pairs"],
                "n_zero_trajectories": f_vals["n_zero_trajectories"],
            })

    csv_path = output_dir / "trajectory_consistency.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved trajectory consistency CSV to {csv_path}")

    # --- Save per-turn stability as CSV ---
    # Rows: (persona, turn, feature), Columns: mean, sd, cv
    stability_rows = []
    for persona, persona_results in results.items():
        mean = persona_results["per_turn_stability"]["mean"]
        sd = persona_results["per_turn_stability"]["sd"]
        cv = persona_results["per_turn_stability"]["cv"]
        for turn_idx in range(mean.shape[0]):
            for f_idx, f_name in enumerate(feature_names):
                stability_rows.append({
                    "persona": persona,
                    "turn": turn_idx + 1,
                    "feature": f_name,
                    "mean": float(mean[turn_idx, f_idx]),
                    "sd": float(sd[turn_idx, f_idx]),
                    "cv": float(cv[turn_idx, f_idx]),
                })

    stability_path = output_dir / "per_turn_stability.csv"
    with open(stability_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stability_rows[0].keys())
        writer.writeheader()
        writer.writerows(stability_rows)
    print(f"Saved per-turn stability CSV to {stability_path}")

    # --- Save full results as JSON (numpy arrays converted to lists) ---
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    json_path = output_dir / "consistency_results.json"
    with open(json_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"Saved full results JSON to {json_path}")


if __name__ == "__main__":
    FEATURES_PATH = "./modules/persona_validation/output/extracted_features.json"

    print(f"Loading features from {FEATURES_PATH}...")
    with open(FEATURES_PATH, "r") as f:
        data = json.load(f)

    conversations_by_persona = data["by_persona"]
    print(f"Loaded {data['metadata']['total_conversations']} conversations across {len(conversations_by_persona)} personas")

    results = calculate_consistency_metrics(conversations_by_persona)
    save_results(results, output_dir="./modules/persona_validation/output/persona_consistency")