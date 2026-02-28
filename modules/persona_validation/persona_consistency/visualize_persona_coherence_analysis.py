"""
Persona Coherence Visualizations

Two visualizations for conversation trajectory coherence analysis:
    1. Bar chart with error bars: average BERTScore F1 scores
    2. Multi-line chart: relevance to initial utterance across turns with error bands

Usage:
    from coherence_visualizations import plot_coherence_results
    plot_coherence_results(persona_coherence_results, output_dir="./output/plots")
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Match your existing style

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
    """Return persona names in consistent order."""
    return [p for p in PERSONA_ORDER if p in results]


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_average_relevance_scores(
    results: dict,
    output_dir: str = "./output/plots",
):
    """
    Bar chart with error bars showing average BERTScore F1 across personas.
    
    Higher scores = more coherent conversation trajectory (consecutive
    assistant responses stay on-topic).
    """
    output_dir = _ensure_dir(output_dir)
    personas = _get_personas(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(personas))
    
    means = [
        results[persona]["average_relevance_scores"]["avg"]
        for persona in personas
    ]
    stds = [
        results[persona]["average_relevance_scores"]["sd"]
        for persona in personas
    ]
    colors = [PERSONA_COLORS[persona] for persona in personas]
    
    bars = ax.bar(
        x, means, 
        yerr=stds,
        color=colors,
        edgecolor="white", 
        linewidth=0.5,
        capsize=5,
        error_kw={'linewidth': 1.5, 'ecolor': 'black'}
    )
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        # Label for mean value (to the left of error bar, above bar)
        ax.text(
            i - 0.175, mean + std + 0.01,
            f'{mean:.3f}',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )
        # Label for SD (to the right of error bar, above bar)
        ax.text(
            i + 0.175, mean + std + 0.01,
            f'±{std:.3f}',
            ha='center', va='bottom',
            fontsize=8, fontweight='bold', alpha=0.9
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(
        [p.replace("_", " ").title() for p in personas],
        fontsize=10, fontweight="bold",
    )

    all_values = means + list(np.array(means) - np.array(stds)) + list(np.array(means) + np.array(stds))
    y_min = max(0, min(all_values) - 0.05)
    y_max = min(1.0, max(all_values) + 0.10)
    
    # Color persona labels
    for p_idx, persona in enumerate(personas):
        ax.get_xticklabels()[p_idx].set_color(PERSONA_COLORS[persona])
    
    ax.set_ylabel("Average BERTScore F1 (Baseline-rescaled)", fontsize=11)
    ax.set_title(
        "Average BERTScore of Sequential LLM Utterance Pairs in Conversations by Persona\n"
        "(average token-wise similarity between consecutive LLM turns)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    #ax.set_ylim(0, max([m + s for m, s in zip(means, stds)]) * 1.20)  # Extra space for labels
    ax.set_ylim(y_min, 1.0)
    
    plt.tight_layout()
    path = output_dir / "coherence_average_relevance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_relevance_to_initial_utterance(
    results: dict,
    output_dir: str = "./output/plots",
):
    """
    Line chart showing how BERTScore relevance to the initial user message 
    changes across the 4 turns of conversation, with shaded error bands.
    
    Decreasing trend = conversation drifts from original topic.
    Stable trend = persona maintains focus on initial query.
    """
    output_dir = _ensure_dir(output_dir)
    personas = _get_personas(results)
    
    n_personas = len(personas)
    n_cols = 2
    n_rows = int(np.ceil(n_personas / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten()
    
    turns = np.arange(3)
    turn_labels = ["T1", "T2", "T3"]
    
    for p_idx, persona in enumerate(personas):
        ax = axes[p_idx]
        
        means = [
            results[persona]["average_relevance_to_initial_utterance"][f"turn_{t}"]["avg"]
            for t in range(1,4)
        ]
        stds = [
            results[persona]["average_relevance_to_initial_utterance"][f"turn_{t}"]["sd"]
            for t in range(1,4)
        ]
        
        # Plot line with markers
        ax.plot(
            turns, means,
            marker="o", markersize=8,
            linestyle="-",
            linewidth=2.5,
            color=PERSONA_COLORS[persona],
            alpha=0.9,
        )
        
        # Add shaded error band (mean ± 1 SD)
        ax.fill_between(
            turns,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=PERSONA_COLORS[persona],
            alpha=0.3,
        )
        
        # Calculate y-axis range for this persona
        all_values = means + list(np.array(means) - np.array(stds)) + list(np.array(means) + np.array(stds))
        y_min = max(0, min(all_values) - 0.05)
        y_max = min(1.0, max(all_values) + 0.10)  # Extra space for labels
        
        # Add value labels at each point
        for t, (mean, std) in enumerate(zip(means, stds)):
            # Mean value label (above point)
            ax.text(
                t, mean + 0.02,
                f'{mean:.3f}',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold',
            )
            # SD label (below point)
            ax.text(
                t, mean - 0.02,
                f'±{std:.3f}',
                ha='center', va='top',
                fontsize=7,
                alpha=0.9
            )
        
        ax.set_xticks(turns)
        ax.set_xticklabels(turn_labels)
        ax.set_xlabel("Turn", fontsize=9)
        ax.set_ylabel("BERTScore F1 (Baseline-rescaled)", fontsize=9)
        ax.set_title(
            persona.replace("_", " ").title(),
            fontsize=11, fontweight="bold",
            color=PERSONA_COLORS[persona],
        )
        ax.grid(alpha=0.3)
        ax.set_ylim(y_min, y_max)  # Dynamic y-axis for each persona
    
    # Hide unused subplots
    for idx in range(n_personas, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(
        "BERTScore Similarity From Initial to Current LLM Utterance Across Conversation Turns\n"
        "(how well LLM responses stay on-topic)",
        fontsize=14, fontweight="bold", y=1.00,
    )
    plt.tight_layout()
    path = output_dir / "coherence_initial_relevance_by_turn.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_coherence_results(
    results: dict,
    output_dir: str = "./output/plots",
):
    """
    Generate both coherence visualizations.
    
    Args:
        results: Output from evaluate_persona_coherence (persona_coherence_results)
        output_dir: Directory to save plots
    """
    print("\nGenerating coherence visualizations...")
    plot_average_relevance_scores(results, output_dir)
    plot_relevance_to_initial_utterance(results, output_dir)
    print("Done.")


# ── Standalone execution ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load your results
    with open("./modules/persona_validation/output/persona_consistency/persona_coherence_results_no_rescale.json", "r") as f:
        results = json.load(f)
    
    plot_coherence_results(results, output_dir="./modules/persona_validation/output/persona_consistency/plots")