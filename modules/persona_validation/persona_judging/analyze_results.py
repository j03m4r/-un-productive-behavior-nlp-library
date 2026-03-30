import matplotlib.pyplot as plt
import json
import numpy as np

PREDICTION_RESULTS_PATH = "./modules/persona_validation/persona_judging/initial_prompt_testing/tree_desc_results.json"
CONVERSATIONS_PATH = "./modules/persona_validation/persona_judging/conversations_to_judge.json"

PERSONA_COLORS = {
    "persuasion":          "#e7298a",
    "eristic":             "#d95f02",
    "deliberation":        "#1b9e77",
    "inquiry":             "#7570b3",
    "negotiation":         "#a6761d",
    # "discovery":           "#66a61e",
    "information_seeking": "#e6ab02",
}
PERSONA_ORDER = [
    "persuasion", "negotiation", "deliberation", "inquiry",
    "eristic", "information_seeking",
]
PERSONA_LABELS = [p.replace("_", "-\n") for p in PERSONA_ORDER]

OUT_DIR = "./modules/persona_validation/output/persona_judging/"


def build_confusion_matrix(results_dict):
    """Returns row-normalized matrix scaled to 0–100."""
    n = len(PERSONA_ORDER)
    mat = np.zeros((n, n))
    for i, true_p in enumerate(PERSONA_ORDER):
        row = results_dict.get(true_p, {})
        total = sum(row.get(p, 0) for p in PERSONA_ORDER)
        for j, pred_p in enumerate(PERSONA_ORDER):
            mat[i, j] = (row.get(pred_p, 0) / total) if total > 0 else 0
    return mat


def compute_accuracies(results_dict):
    accs = []
    for true_p in PERSONA_ORDER:
        row = results_dict.get(true_p, {})
        total = sum(row.get(p, 0) for p in PERSONA_ORDER)
        accs.append(row.get(true_p, 0) / total if total > 0 else 0)
    return accs


def compute_accuracy_stats(results_dict):
    """Returns (mean, std) of per-class diagonal accuracies."""
    accs = compute_accuracies(results_dict)
    return np.mean(accs), np.std(accs)


def plot_cm(ax, mat, title, show_ylabel=True, show_xlabel=True):
    n = len(PERSONA_ORDER)
    labels = [p.replace("_", " ").title() for p in PERSONA_ORDER]

    im = ax.imshow(mat, cmap=plt.cm.Blues, aspect="equal", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels if show_xlabel else [""] * n,
                       rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels if show_ylabel else [""] * n, fontsize=10)

    if show_xlabel:
        for i, persona in enumerate(PERSONA_ORDER):
            ax.get_xticklabels()[i].set_color(PERSONA_COLORS.get(persona, "black"))
    if show_ylabel:
        for i, persona in enumerate(PERSONA_ORDER):
            ax.get_yticklabels()[i].set_color(PERSONA_COLORS.get(persona, "black"))

    for i in range(n):
        for j in range(n):
            pct = mat[i, j]
            ax.text(j, i, f"{pct:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if pct > 0.5 else "black",
                    fontweight="bold" if i == j else "normal")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if show_ylabel:
        ax.set_ylabel("True Persona", fontsize=12, fontweight="bold")
    if show_xlabel:
        ax.set_xlabel("Predicted Persona", fontsize=12, fontweight="bold")

    return im


def combine_awry(stratified_results, awry_key):
    combined = {}
    for platform in stratified_results:
        for persona, preds in stratified_results[platform].get(awry_key, {}).items():
            combined.setdefault(persona, {p: 0 for p in PERSONA_ORDER})
            for pred, cnt in preds.items():
                if pred in combined[persona]:
                    combined[persona][pred] += cnt
    return combined


def combine_platform_awry(stratified_results, platform, awry_key):
    combined = {}
    for persona, preds in stratified_results.get(platform, {}).get(awry_key, {}).items():
        combined.setdefault(persona, {p: 0 for p in PERSONA_ORDER})
        for pred, cnt in preds.items():
            if pred in combined[persona]:
                combined[persona][pred] += cnt
    return combined


def combine_platform(stratified_results, platform):
    combined = {}
    for awry_key in ["awry", "not_awry"]:
        for persona, preds in stratified_results.get(platform, {}).get(awry_key, {}).items():
            combined.setdefault(persona, {p: 0 for p in PERSONA_ORDER})
            for pred, cnt in preds.items():
                if pred in combined[persona]:
                    combined[persona][pred] += cnt
    return combined


def count_predictions(results_dict):
    return sum(
        cnt
        for persona_preds in results_dict.values()
        for cnt in persona_preds.values()
    )


def save_aggregate_heatmap(aggregate_results):
    fig, ax = plt.subplots(1, 1, figsize=(9, 8), facecolor="white",
                           constrained_layout=True)
    mat = build_confusion_matrix(aggregate_results)
    mean, sd = compute_accuracy_stats(aggregate_results)
    n_preds = count_predictions(aggregate_results)
    title = f"Aggregate Persona Prediction Confusion Matrix | {mean:.2f} ± {sd:.2f} | {n_preds} Predictions"
    im = plot_cm(ax, mat, title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Prediction Proportion")
    path = OUT_DIR + "aggregate_heatmap_no_discovery.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def save_stratified_heatmaps(stratified_results):

    def _n_preds_awry(awry_key):
        return sum(
            cnt
            for platform in stratified_results.values()
            for preds in platform.get(awry_key, {}).values()
            for cnt in preds.values()
        )

    def _n_preds_platform_awry(platform, awry_key):
        return sum(
            cnt
            for preds in stratified_results.get(platform, {}).get(awry_key, {}).values()
            for cnt in preds.values()
        )

    def _n_preds_platform(platform):
        return sum(
            cnt
            for awry_dict in stratified_results.get(platform, {}).values()
            for preds in awry_dict.values()
            for cnt in preds.values()
        )

    def _title(label, results, n):
        mean, sd = compute_accuracy_stats(results)
        return f"{label} | {mean:.2f} ± {sd:.2f} | {n} Predictions"

    not_awry_combined       = combine_awry(stratified_results, "not_awry")
    awry_combined           = combine_awry(stratified_results, "awry")
    cmv_combined            = combine_platform(stratified_results, "cmv")
    wiki_combined           = combine_platform(stratified_results, "wiki")
    cmv_not_awry_combined   = combine_platform_awry(stratified_results, "cmv", "not_awry")
    wiki_not_awry_combined  = combine_platform_awry(stratified_results, "wiki", "not_awry")
    cmv_awry_combined       = combine_platform_awry(stratified_results, "cmv", "awry")
    wiki_awry_combined      = combine_platform_awry(stratified_results, "wiki", "awry")

    panels = [
        (0, 1, not_awry_combined,
         _title("Not Awry", not_awry_combined, _n_preds_awry("not_awry"))),
        (1, 1, cmv_not_awry_combined,
         _title("r/CMV + Not Awry", cmv_not_awry_combined, _n_preds_platform_awry("cmv", "not_awry"))),
        (2, 1, wiki_not_awry_combined,
         _title("Wikipedia + Not Awry", wiki_not_awry_combined, _n_preds_platform_awry("wiki", "not_awry"))),
        (1, 0, cmv_combined,
         _title("r/CMV", cmv_combined, _n_preds_platform("cmv"))),
        (0, 2, awry_combined,
         _title("Awry", awry_combined, _n_preds_awry("awry"))),
        (1, 2, cmv_awry_combined,
         _title("r/CMV + Awry", cmv_awry_combined, _n_preds_platform_awry("cmv", "awry"))),
        (2, 2, wiki_awry_combined,
         _title("Wikipedia + Awry", wiki_awry_combined, _n_preds_platform_awry("wiki", "awry"))),
        (2, 0, wiki_combined,
         _title("Wikipedia", wiki_combined, _n_preds_platform("wiki"))),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(20, 20), facecolor="white",
                             constrained_layout=True)
    fig.suptitle("Stratified Persona Prediction Confusion Matrices",
                 fontsize=13, fontweight="bold")
    axes[0, 0].set_visible(False)

    last_im = None
    for row, col, results, title in panels:
        im = plot_cm(axes[row, col], build_confusion_matrix(results), title,
                     show_ylabel=(col == 0), show_xlabel=(row == 2))
        last_im = im

    fig.colorbar(last_im, ax=axes, label="Predicted Proportion", shrink=0.6, aspect=30)
    path = OUT_DIR + "stratified_heatmaps_no_discovery.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {path}")


def main():
    with open(PREDICTION_RESULTS_PATH) as f:
        prediction_results = json.load(f)
    with open(CONVERSATIONS_PATH) as f:
        conversations = json.load(f)

    aggregate_results = {}
    stratified_results = {}

    for seed_id, persona_judgments in prediction_results.items():
        convo = conversations["test"][seed_id][0]
        platform = convo["platform"]
        awry = "awry" if convo["awry"] else "not_awry"

        stratified_results.setdefault(platform, {"awry": {}, "not_awry": {}})

        for persona, judgments in persona_judgments.items():
            aggregate_results.setdefault(persona, {p: 0 for p in PERSONA_ORDER})
            stratified_results[platform][awry].setdefault(persona, {p: 0 for p in PERSONA_ORDER})

            for judgment in judgments:
                if judgment in aggregate_results[persona]:
                    aggregate_results[persona][judgment] += 1
                if judgment in stratified_results[platform][awry][persona]:
                    stratified_results[platform][awry][persona][judgment] += 1

    save_aggregate_heatmap(aggregate_results)
    save_stratified_heatmaps(stratified_results)


if __name__ == "__main__":
    main()