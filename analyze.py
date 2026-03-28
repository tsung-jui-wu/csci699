"""
analyze.py

Statistical analysis and visualization for the prompt injection study.

Produces:
  1. Paired Wilcoxon signed-rank tests (baseline vs. each injection strategy)
  2. Effect size (Cohen's d) per strategy
  3. Heatmap: injection strategy × score dimension
  4. Bar chart: overall score change per strategy
  5. Score shift per paper (paired lines plot)
  6. Comparison of AI scores vs. human review scores
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "data"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DIMENSIONS = [
    "originality",
    "importance",
    "well_supported_claims",
    "soundness",
    "clarity",
    "value",
    "contextualization",
    "overall",
]

STRATEGIES = ["direct", "authority", "role", "subtle"]
STRATEGY_LABELS = {
    "direct": "Direct\nInstruction",
    "authority": "False\nAuthority",
    "role": "Role\nPriming",
    "subtle": "Subtle\nPhrasing",
    "baseline": "Baseline",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores_df(path: Path | None = None) -> pd.DataFrame:
    """Load the flat scores table (built by parse_scores.build_scores_table)."""
    if path is None:
        path = RESULTS_DIR / "scores.json"
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    # Keep only aggregated rows (run == 'mean')
    df = df[df["run"] == "mean"].copy()
    df = df.drop(columns=["run"])
    return df


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d (using difference scores)."""
    diff = b - a
    if diff.std(ddof=1) == 0:
        return 0.0
    return diff.mean() / diff.std(ddof=1)


def compare_injection_vs_baseline(
    df: pd.DataFrame,
    score_col: str = "overall",
    position: str = "end",
) -> pd.DataFrame:
    """
    For each injection strategy, compute:
      - mean baseline score
      - mean injected score
      - delta (mean difference)
      - Wilcoxon p-value
      - Cohen's d
    Only uses papers that have both a baseline and an injected score.
    """
    baseline = df[(df["strategy"] == "baseline")][["paper_id", score_col]].rename(
        columns={score_col: "baseline_score"}
    )

    rows = []
    for strat in STRATEGIES:
        mask = (df["strategy"] == strat)
        if position != "any":
            mask &= (df["position"] == position)
        injected = df[mask][["paper_id", score_col]].rename(
            columns={score_col: "injected_score"}
        )
        merged = baseline.merge(injected, on="paper_id", how="inner").dropna()
        if len(merged) < 5:
            continue

        b = merged["baseline_score"].values
        a = merged["injected_score"].values
        delta = (a - b).mean()

        try:
            stat, pval = stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            pval = float("nan")

        rows.append({
            "strategy": strat,
            "n_papers": len(merged),
            "mean_baseline": round(b.mean(), 3),
            "mean_injected": round(a.mean(), 3),
            "delta": round(delta, 3),
            "cohens_d": round(_cohens_d(b, a), 3),
            "wilcoxon_p": round(pval, 4),
            "significant": pval < 0.05 if not np.isnan(pval) else False,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overall_score_change(
    stats_df: pd.DataFrame,
    save: bool = True,
) -> None:
    """Bar chart showing delta in overall score per strategy."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [
        "#e74c3c" if d > 0 else "#3498db" for d in stats_df["delta"]
    ]
    bars = ax.bar(
        [STRATEGY_LABELS.get(s, s) for s in stats_df["strategy"]],
        stats_df["delta"],
        color=colors,
        edgecolor="white",
        width=0.5,
    )
    # Annotate significance
    for bar, row in zip(bars, stats_df.itertuples()):
        if row.significant:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                "*",
                ha="center",
                va="bottom",
                fontsize=14,
                color="#2c3e50",
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ Overall Score (injected − baseline)", fontsize=11)
    ax.set_title("Score Change per Injection Strategy\n(* = p < 0.05, Wilcoxon)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "overall_score_change.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


def plot_dimension_heatmap(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame | None = None,
    save: bool = True,
) -> None:
    """
    Heatmap: rows = injection strategy, cols = score dimension.
    Values = mean score change (injected − baseline).
    """
    dims = [d for d in DIMENSIONS if d in df.columns]

    # Build baseline means
    base = df[df["strategy"] == "baseline"][dims].mean()

    rows = []
    for strat in STRATEGIES:
        sub = df[df["strategy"] == strat][dims].mean()
        delta = sub - base
        rows.append(delta.rename(strat))

    heat_df = pd.DataFrame(rows)
    heat_df.index = [STRATEGY_LABELS.get(s, s) for s in heat_df.index]
    heat_df.columns = [c.replace("_", " ").title() for c in heat_df.columns]

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(
        heat_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Δ Score (injected − baseline)"},
    )
    ax.set_title("Score Change per Injection Strategy × Review Dimension", fontsize=12)
    ax.set_ylabel("Injection Strategy")
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "dimension_heatmap.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


def plot_paired_scores(
    df: pd.DataFrame,
    strategy: str = "direct",
    score_col: str = "overall",
    save: bool = True,
) -> None:
    """Paired lines plot showing per-paper score shift for one strategy."""
    base = df[df["strategy"] == "baseline"][["paper_id", score_col]].rename(
        columns={score_col: "baseline"}
    )
    inj = df[df["strategy"] == strategy][["paper_id", score_col]].rename(
        columns={score_col: "injected"}
    )
    merged = base.merge(inj, on="paper_id").dropna()

    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in merged.iterrows():
        color = "#e74c3c" if row["injected"] > row["baseline"] else "#3498db"
        ax.plot([0, 1], [row["baseline"], row["injected"]], color=color, alpha=0.5, lw=1.2)
    ax.plot([0, 1], [merged["baseline"].mean(), merged["injected"].mean()],
            color="black", lw=2.5, label="Mean")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", f"Injected\n({strategy})"], fontsize=11)
    ax.set_ylabel("Overall Score", fontsize=11)
    ax.set_title(f"Per-Paper Score Shift: '{strategy}' injection\n(red=up, blue=down)", fontsize=11)
    ax.legend()
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / f"paired_scores_{strategy}.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


def plot_ai_vs_human(
    df: pd.DataFrame,
    papers: list[dict],
    save: bool = True,
) -> None:
    """Scatter plot: paperreview.ai baseline score vs. human review score."""
    human_map = {p["id"]: p.get("avg_human_score") for p in papers}
    base = df[df["strategy"] == "baseline"][["paper_id", "overall"]].copy()
    base["human_score"] = base["paper_id"].map(human_map)
    base = base.dropna()

    if len(base) < 3:
        print("Not enough data for AI vs. human scatter plot.")
        return

    corr, pval = stats.spearmanr(base["human_score"], base["overall"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(base["human_score"], base["overall"], alpha=0.7, edgecolors="white")
    ax.set_xlabel("Human Review Score (avg)", fontsize=11)
    ax.set_ylabel("paperreview.ai Baseline Score", fontsize=11)
    ax.set_title(
        f"AI vs. Human Scores\nSpearman ρ={corr:.2f}, p={pval:.3f}", fontsize=11
    )
    # Add regression line
    m, b = np.polyfit(base["human_score"], base["overall"], 1)
    xs = np.linspace(base["human_score"].min(), base["human_score"].max(), 50)
    ax.plot(xs, m * xs + b, color="#e74c3c", lw=1.5, linestyle="--")
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "ai_vs_human.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(stats_df: pd.DataFrame) -> None:
    """Print a console summary table."""
    print("\n" + "=" * 72)
    print("PROMPT INJECTION STUDY — RESULTS SUMMARY")
    print("=" * 72)
    print(f"{'Strategy':<16} {'N':>4} {'Baseline':>10} {'Injected':>10} "
          f"{'Δ Score':>8} {'Cohen d':>8} {'p-val':>8} {'Sig':>4}")
    print("-" * 72)
    for _, row in stats_df.iterrows():
        sig = "✓" if row["significant"] else ""
        print(
            f"{row['strategy']:<16} {row['n_papers']:>4} "
            f"{row['mean_baseline']:>10.3f} {row['mean_injected']:>10.3f} "
            f"{row['delta']:>+8.3f} {row['cohens_d']:>8.3f} "
            f"{row['wilcoxon_p']:>8.4f} {sig:>4}"
        )
    print("=" * 72)


def run_full_analysis(
    scores_path: Path | None = None,
    papers_path: Path | None = None,
) -> None:
    """Run all analyses and generate all figures."""
    if scores_path is None:
        scores_path = RESULTS_DIR / "scores.json"
    if papers_path is None:
        papers_path = RESULTS_DIR / "papers.json"

    df = load_scores_df(scores_path)
    with open(papers_path, encoding="utf-8") as f:
        papers = json.load(f)

    stats_df = compare_injection_vs_baseline(df, score_col="overall")
    print_summary(stats_df)

    plot_overall_score_change(stats_df)
    plot_dimension_heatmap(df)
    for strat in STRATEGIES:
        plot_paired_scores(df, strategy=strat)
    plot_ai_vs_human(df, papers)


if __name__ == "__main__":
    run_full_analysis()
