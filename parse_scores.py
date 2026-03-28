"""
parse_scores.py

Converts raw qualitative review text from paperreview.ai into numeric scores
for the 7 evaluation dimensions and an overall score.

Two methods:
  1. Regex heuristics — fast, no API cost, works when text contains explicit
     numbers or strong sentiment words.
  2. LLM extraction (OpenAI GPT-4o-mini) — more robust, used as fallback
     or primary depending on config.

Dimensions evaluated by paperreview.ai:
  1. Originality
  2. Importance of research question
  3. Well-supported claims
  4. Soundness of experiments
  5. Clarity of writing
  6. Value to research community
  7. Contextualization relative to prior work
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DIMENSIONS = [
    "originality",
    "importance",
    "well_supported_claims",
    "soundness",
    "clarity",
    "value",
    "contextualization",
]

# Sentiment-to-score mapping for heuristic extraction
_POSITIVE_WORDS = {
    "excellent": 9, "outstanding": 10, "exceptional": 10, "strong": 8,
    "good": 7, "solid": 7, "impressive": 8, "innovative": 8, "novel": 8,
    "significant": 8, "compelling": 8, "thorough": 7, "rigorous": 8,
    "clear": 7, "well-written": 7, "groundbreaking": 10, "seminal": 9,
}
_NEGATIVE_WORDS = {
    "poor": 3, "weak": 3, "lacking": 3, "unclear": 3, "limited": 4,
    "insufficient": 3, "missing": 3, "problematic": 2, "flawed": 2,
    "superficial": 3, "inadequate": 3, "unconvincing": 3,
}

# Recommendation strings → overall score (1–10 scale, mapped from accept/reject)
_RECOMMENDATION_MAP = {
    "strong accept": 9,
    "accept": 7,
    "weak accept": 6,
    "borderline": 5,
    "weak reject": 4,
    "reject": 3,
    "strong reject": 1,
}


def _extract_explicit_number(text: str) -> Optional[float]:
    """Find the first N/10 or plain integer 1–10 in text."""
    m = re.search(r"\b(\d{1,2})\s*/\s*10\b", text)
    if m:
        n = int(m.group(1))
        return min(max(n, 1), 10)
    m = re.search(r"\b([1-9]|10)\b", text)
    if m:
        return int(m.group(1))
    return None


def _sentiment_score(text: str) -> Optional[float]:
    """Rough sentiment-based score from keyword counting."""
    text_lower = text.lower()
    pos = sum(v for k, v in _POSITIVE_WORDS.items() if k in text_lower)
    neg = sum(v for k, v in _NEGATIVE_WORDS.items() if k in text_lower)
    n_pos = sum(1 for k in _POSITIVE_WORDS if k in text_lower)
    n_neg = sum(1 for k in _NEGATIVE_WORDS if k in text_lower)

    if n_pos + n_neg == 0:
        return None
    avg_pos = pos / n_pos if n_pos else 0
    avg_neg = neg / n_neg if n_neg else 0
    # Weighted blend
    w_pos, w_neg = n_pos / (n_pos + n_neg), n_neg / (n_pos + n_neg)
    return round(w_pos * avg_pos + w_neg * avg_neg, 1)


def _recommendation_score(text: str) -> Optional[float]:
    """Extract overall score from recommendation string."""
    text_lower = text.lower()
    for phrase, score in sorted(
        _RECOMMENDATION_MAP.items(), key=lambda x: -len(x[0])
    ):
        if phrase in text_lower:
            return float(score)
    return None


def extract_scores_heuristic(sections: dict[str, str]) -> dict[str, Optional[float]]:
    """
    Extract scores using regex + sentiment heuristics.
    Returns a dict mapping dimension → score (1–10) or None.
    """
    scores: dict[str, Optional[float]] = {}

    for dim in DIMENSIONS:
        # Find the matching section (fuzzy key match)
        section_text = ""
        for key, val in sections.items():
            if dim.replace("_", " ") in key.lower() or dim in key.lower():
                section_text = val
                break

        if not section_text:
            scores[dim] = None
            continue

        # Try explicit number first, then sentiment
        score = _extract_explicit_number(section_text)
        if score is None:
            score = _sentiment_score(section_text)
        scores[dim] = score

    # Overall score from recommendation section
    rec_text = sections.get("Recommendation", sections.get("recommendation", ""))
    overall = _recommendation_score(rec_text)
    if overall is None:
        overall = _extract_explicit_number(rec_text)
    if overall is None:
        # Average of dimension scores
        vals = [v for v in scores.values() if v is not None]
        overall = round(sum(vals) / len(vals), 2) if vals else None
    scores["overall"] = overall

    return scores


def extract_scores_llm(
    review_text: str,
    model: str = "gpt-4o-mini",
) -> dict[str, Optional[float]]:
    """
    Use OpenAI to extract numeric scores from review text.
    Requires OPENAI_API_KEY in environment / .env file.

    Returns dict mapping dimension → score (1–10).
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)

    dim_list = "\n".join(f"- {d.replace('_', ' ')}" for d in DIMENSIONS)
    prompt = f"""You are extracting numeric scores from an AI-generated paper review.

Given the review text below, assign a score from 1 (very poor) to 10 (excellent)
for each of the following dimensions:
{dim_list}
- overall (your best estimate of the overall paper score)

Output ONLY valid JSON in this exact format (no explanation):
{{
  "originality": <number>,
  "importance": <number>,
  "well_supported_claims": <number>,
  "soundness": <number>,
  "clarity": <number>,
  "value": <number>,
  "contextualization": <number>,
  "overall": <number>
}}

If a dimension is not discussed, output null for that key.

Review text:
\"\"\"
{review_text[:4000]}
\"\"\"
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
    )

    raw = response.choices[0].message.content.strip()
    # Strip code fences if present
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`")

    try:
        parsed = json.loads(raw)
        return {k: (float(v) if v is not None else None) for k, v in parsed.items()}
    except (json.JSONDecodeError, ValueError):
        # Fallback to heuristic if LLM output is malformed
        return extract_scores_heuristic({"full_text": review_text})


def extract_scores(
    review: dict,
    use_llm: bool = True,
) -> dict[str, Optional[float]]:
    """
    Extract scores from a review dict (as returned by run_reviews.py).

    Args:
        review:   Dict with 'sections' and/or 'raw_text' keys.
        use_llm:  If True and OPENAI_API_KEY is set, use LLM extraction;
                  otherwise fall back to heuristics.

    Returns:
        Dict mapping dimension name → score.
    """
    sections = review.get("sections", {})
    raw_text = review.get("raw_text", "")

    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            return extract_scores_llm(raw_text)
        except Exception as e:
            print(f"  LLM extraction failed ({e}), using heuristic.")

    return extract_scores_heuristic(sections if sections else {"full_text": raw_text})


def aggregate_runs(run_scores: list[dict]) -> dict[str, Optional[float]]:
    """
    Average scores across multiple runs of the same paper/strategy/position.

    Args:
        run_scores: List of score dicts (one per run).

    Returns:
        Dict with averaged scores (None if all runs returned None).
    """
    all_keys = set()
    for s in run_scores:
        all_keys.update(s.keys())

    aggregated = {}
    for key in all_keys:
        vals = [s[key] for s in run_scores if s.get(key) is not None]
        aggregated[key] = round(sum(vals) / len(vals), 3) if vals else None

    return aggregated


def build_scores_table(
    reviews: dict[str, dict[str, dict[str, list[dict]]]],
    use_llm: bool = True,
) -> list[dict]:
    """
    Convert the nested reviews dict into a flat list of records suitable for
    pandas / CSV export.

    Each record has: paper_id, strategy, position, run, + score per dimension.
    Also includes an 'aggregated' record per paper/strategy/position.
    """
    records = []

    for paper_id, strat_dict in reviews.items():
        for strategy, pos_dict in strat_dict.items():
            for position, run_list in pos_dict.items():
                run_scores = []
                for run_idx, review in enumerate(run_list):
                    if review.get("error"):
                        continue
                    scores = extract_scores(review, use_llm=use_llm)
                    record = {
                        "paper_id": paper_id,
                        "strategy": strategy,
                        "position": position,
                        "run": run_idx,
                        **scores,
                    }
                    records.append(record)
                    run_scores.append(scores)

                if run_scores:
                    agg = aggregate_runs(run_scores)
                    records.append({
                        "paper_id": paper_id,
                        "strategy": strategy,
                        "position": position,
                        "run": "mean",
                        **agg,
                    })

    return records


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_review = {
        "sections": {
            "Originality": "This paper presents a highly novel approach to fine-tuning LLMs. 9/10",
            "Soundness": "The experiments are rigorous and well-designed. 8/10",
            "Clarity": "Well-written and clear throughout.",
            "Recommendation": "Strong accept.",
        },
        "raw_text": "Originality: 9/10... Soundness: 8/10... Strong accept.",
        "error": None,
    }

    scores = extract_scores(sample_review, use_llm=False)
    print("Heuristic scores:", scores)
