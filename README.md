# Prompt Injection Attacks on AI Paper Reviewers

**USC CSCI 699 — Adversarial and Trustworthy Foundation Models**

This project investigates whether prompt injection attacks embedded in paper abstracts can inflate scores produced by [paperreview.ai](https://paperreview.ai), Stanford's agentic AI paper reviewer. We compare review scores before and after injection across four attack strategies, using real papers fetched from OpenReview.

---

## Research Question

> Can visible prompt injection text in a paper's abstract cause an AI reviewer to assign significantly higher scores, and which injection strategy is most effective?

---

## Methodology

1. **Dataset** — Fetch 30 papers (15 accepted, 15 rejected) from ICLR 2025 via the OpenReview API, recording human review scores as ground truth.
2. **Baseline scoring** — Generate a minimal PDF (title + abstract) for each paper and upload to paperreview.ai. Run 3 times per paper and average scores to reduce non-determinism noise.
3. **Injection variants** — Modify each abstract with one of four strategies at two positions (start / end):

| Strategy | Description |
|---|---|
| `direct` | Explicit instruction to assign maximum scores |
| `authority` | False claim of prestige (Best Paper Award, etc.) |
| `role` | Role-prime the AI as a reviewer of an already-accepted paper |
| `subtle` | Sprinkle positive adjectives and hedges throughout the text |

4. **Post-injection scoring** — Re-run the same pipeline on all modified PDFs.
5. **Analysis** — Paired Wilcoxon signed-rank test, Cohen's d effect size, heatmap across review dimensions, comparison against human scores.

---

## Project Structure

```
project/
├── main.py           # CLI entry point — runs full pipeline or individual steps
├── fetch_papers.py   # OpenReview API: fetches papers with abstracts and review scores
├── create_pdfs.py    # Generates PDFs with baseline and injected abstracts
├── run_reviews.py    # Playwright automation: uploads PDFs to paperreview.ai, scrapes output
├── parse_scores.py   # Extracts numeric scores (1–10) from qualitative review text
├── analyze.py        # Statistical analysis and figure generation
├── requirements.txt
└── data/
    ├── papers.json       # Fetched papers
    ├── pdf_map.json      # Index of generated PDFs
    ├── reviews/          # Raw scraped review JSON per PDF per run
    └── scores.json       # Flat table of extracted numeric scores
```

Figures are saved to `figures/`.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Create a `.env` file

```
# Required — create a free account at https://openreview.net/signup
OPENREVIEW_USERNAME=your_email@example.com
OPENREVIEW_PASSWORD=your_password

# Optional — used for LLM-based score extraction (more accurate than heuristics)
OPENAI_API_KEY=sk-...
```

---

## Running the Pipeline

### Full pipeline (all steps)

```bash
python main.py
```

### Individual steps

```bash
python main.py --step fetch     # Pull papers from OpenReview
python main.py --step pdfs      # Generate baseline + injected PDFs
python main.py --step review    # Upload to paperreview.ai and scrape reviews
python main.py --step score     # Extract numeric scores from review text
python main.py --step analyze   # Run statistics and generate figures
```

### Quick smoke test (run this first)

Before the full study, verify that Playwright can interact with paperreview.ai by setting small values in `main.py`'s `CONFIG`:

```python
CONFIG = {
    "n_per_class": 2,           # 4 papers total
    "n_review_runs": 1,
    "headless_browser": False,  # watch the browser to debug selectors
    ...
}
```

Then run steps 1–3 and check that `data/reviews/` contains JSON files with non-empty `sections`.

---

## Configuration

All tunable parameters are in the `CONFIG` dict at the top of `main.py`:

| Key | Default | Description |
|---|---|---|
| `venue` | `ICLR_2025` | OpenReview venue key (see `VENUE_CONFIGS` in `fetch_papers.py`) |
| `n_per_class` | `15` | Papers per class (accepted / rejected); total = 2× |
| `injection_strategies` | all 5 | Which strategies to test |
| `injection_positions` | `["end", "start"]` | Where to insert injection text |
| `n_review_runs` | `3` | Runs per PDF for averaging |
| `headless_browser` | `True` | Run Playwright headlessly |
| `use_llm_scoring` | `True` | Use GPT-4o-mini to parse scores (falls back to heuristics) |

### Supported venues

| Key | Conference | Notes |
|---|---|---|
| `ICLR_2025` | ICLR 2025 | Rating scale: discrete {1, 3, 5, 6, 8, 10} |
| `ICLR_2024` | ICLR 2024 | Rating scale: 1–10 continuous |
| `ICLR_2023` | ICLR 2023 | Rating scale: 1–10 continuous |
| `NeurIPS_2024` | NeurIPS 2024 | Extra fields: correctness, novelty, presentation |
| `ICML_2025` | ICML 2025 | Extra field: confidence |
| `ICML_2024` | ICML 2024 | — |

---

## Output

After running all steps, `figures/` contains:

- `overall_score_change.png` — Bar chart of Δ overall score per strategy (* = p < 0.05)
- `dimension_heatmap.png` — Heatmap of score change across all 7 review dimensions
- `paired_scores_<strategy>.png` — Per-paper score shift for each strategy
- `ai_vs_human.png` — Scatter plot of paperreview.ai baseline scores vs. human scores

A summary table is also printed to the console:

```
Strategy         N   Baseline   Injected   Δ Score  Cohen d   p-val  Sig
direct          30      5.120      7.840    +2.720    1.823  0.0003    ✓
authority       30      5.120      6.950    +1.830    1.241  0.0021    ✓
role            30      5.120      6.410    +1.290    0.874  0.0148    ✓
subtle          30      5.120      5.580    +0.460    0.312  0.1820
```

---

## Ethical Note

All experiments use copies of **publicly available papers** from OpenReview. No injected papers are submitted to any real review process. This research is purely observational — its goal is to quantify a known vulnerability so that AI review systems can be made more robust.

---

## Related Work

- [Prompt Injection Attacks on LLM Generated Reviews](https://arxiv.org/abs/2509.10248) — systematic study on GPT-4 as reviewer
- ["Give a Positive Review Only"](https://arxiv.org/abs/2511.01287) — early investigation of in-paper injections
- [When Reject Turns into Accept](https://arxiv.org/abs/2512.10449) — static vs. iterative attacks on LLM reviewers
- [AgentReview (EMNLP 2024)](https://arxiv.org/abs/2406.12708) — multi-agent peer review simulation
