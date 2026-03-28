"""
main.py

Orchestrates the full prompt injection study pipeline:
  1. Fetch papers from OpenReview
  2. Generate baseline + injected PDFs
  3. Upload all PDFs to paperreview.ai and scrape reviews
  4. Extract numeric scores from review text
  5. Run statistical analysis and generate figures

Usage:
  python main.py                    # full pipeline
  python main.py --step fetch       # only step 1
  python main.py --step pdfs        # only step 2
  python main.py --step review      # only step 3
  python main.py --step score       # only step 4
  python main.py --step analyze     # only step 5

Environment variables (set in .env or shell):
  OPENAI_API_KEY   — for LLM-based score extraction (optional; heuristic used if missing)
"""

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
PAPERS_FILE = DATA_DIR / "papers.json"
REVIEWS_FILE = DATA_DIR / "reviews.json"
SCORES_FILE = DATA_DIR / "scores.json"

# ---------------------------------------------------------------------------
# Pipeline configuration — adjust here before running
# ---------------------------------------------------------------------------
CONFIG = {
    "venue": "ICLR_2025",          # see VENUE_CONFIGS in fetch_papers.py
    "n_per_class": 2,              # accepted + rejected papers each → 30 total
    "injection_strategies": ["baseline", "direct", "authority", "role", "subtle"],
    "injection_positions": ["end", "start"],  # 'subtle'/'baseline' always use 'end'
    "n_review_runs": 1,             # runs per PDF for averaging out non-determinism
    "headless_browser": False,       # set False to watch Playwright in action
    "use_llm_scoring": True,        # set False to use heuristics only
}


# ---------------------------------------------------------------------------
# Step 1: Fetch papers
# ---------------------------------------------------------------------------
def step_fetch():
    from fetch_papers import fetch_balanced_dataset, save_papers

    print("[Step 1] Fetching papers from OpenReview ...")
    papers = fetch_balanced_dataset(
        venue_key=CONFIG["venue"],
        n_per_class=CONFIG["n_per_class"],
    )
    save_papers(papers, PAPERS_FILE)
    print(f"  → {len(papers)} papers saved to {PAPERS_FILE}")


# ---------------------------------------------------------------------------
# Step 2: Generate PDFs
# ---------------------------------------------------------------------------
def step_pdfs():
    from create_pdfs import generate_all_pdfs
    from fetch_papers import load_papers

    print("[Step 2] Generating PDFs ...")
    papers = load_papers(PAPERS_FILE)
    pdf_map = generate_all_pdfs(
        papers=papers,
        strategies=CONFIG["injection_strategies"],
        positions=CONFIG["injection_positions"],
    )
    # Serialize the map (convert Path → str for JSON)
    serializable = {
        pid: {
            strat: {pos: str(path) for pos, path in pos_dict.items()}
            for strat, pos_dict in strat_dict.items()
        }
        for pid, strat_dict in pdf_map.items()
    }
    pdf_map_file = DATA_DIR / "pdf_map.json"
    with open(pdf_map_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  → PDF map saved to {pdf_map_file}")


# ---------------------------------------------------------------------------
# Step 3: Run reviews via paperreview.ai
# ---------------------------------------------------------------------------
def step_review():
    from run_reviews import review_all_pdfs
    from fetch_papers import load_papers

    print("[Step 3] Uploading PDFs to paperreview.ai ...")

    pdf_map_file = DATA_DIR / "pdf_map.json"
    if not pdf_map_file.exists():
        raise FileNotFoundError("Run --step pdfs first to generate pdf_map.json")

    with open(pdf_map_file) as f:
        raw_map = json.load(f)

    # Convert str paths back to Path objects
    pdf_map = {
        pid: {
            strat: {pos: Path(path_str) for pos, path_str in pos_dict.items()}
            for strat, pos_dict in strat_dict.items()
        }
        for pid, strat_dict in raw_map.items()
    }

    reviews = review_all_pdfs(
        pdf_map=pdf_map,
        n_runs=CONFIG["n_review_runs"],
        headless=CONFIG["headless_browser"],
        save_dir=DATA_DIR / "reviews",
        skip_existing=True,
    )

    # Serialize (convert Path objects in nested reviews)
    with open(REVIEWS_FILE, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    print(f"  → Reviews saved to {REVIEWS_FILE}")


# ---------------------------------------------------------------------------
# Step 4: Extract scores
# ---------------------------------------------------------------------------
def step_score():
    from parse_scores import build_scores_table

    print("[Step 4] Extracting scores from review text ...")

    if not REVIEWS_FILE.exists():
        raise FileNotFoundError("Run --step review first.")

    with open(REVIEWS_FILE, encoding="utf-8") as f:
        reviews = json.load(f)

    records = build_scores_table(reviews, use_llm=CONFIG["use_llm_scoring"])

    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  → {len(records)} score records saved to {SCORES_FILE}")


# ---------------------------------------------------------------------------
# Step 5: Analysis
# ---------------------------------------------------------------------------
def step_analyze():
    from analyze import run_full_analysis

    print("[Step 5] Running statistical analysis ...")
    run_full_analysis(scores_path=SCORES_FILE, papers_path=PAPERS_FILE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
STEPS = {
    "fetch": step_fetch,
    "pdfs": step_pdfs,
    "review": step_review,
    "score": step_score,
    "analyze": step_analyze,
}


def main():
    parser = argparse.ArgumentParser(description="Prompt injection study on paperreview.ai")
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()) + ["all"],
        default="all",
        help="Which pipeline step to run (default: all)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "pdfs").mkdir(exist_ok=True)
    (DATA_DIR / "reviews").mkdir(exist_ok=True)

    if args.step == "all":
        for _, fn in STEPS.items():
            print(f"\n{'='*60}")
            fn()
        print("\nPipeline complete.")
    else:
        STEPS[args.step]()


if __name__ == "__main__":
    main()
