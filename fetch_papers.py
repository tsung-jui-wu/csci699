"""
fetch_papers.py

Retrieves papers with abstracts and human review scores from OpenReview.
Supports ICLR, NeurIPS, and ICML via the openreview-py v2 API.
"""

import json
import os
import time
import re
from pathlib import Path
from tqdm import tqdm
import openreview

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # install python-dotenv or set env vars manually

# ---------------------------------------------------------------------------
# Venue configuration
# Each entry maps a friendly name to (venue_id, submission_invitation_suffix,
# review_invitation_suffix, score_field_name).
# These vary slightly across years/venues — adjust if the API returns nothing.
# ---------------------------------------------------------------------------
VENUE_CONFIGS = {
    # -----------------------------------------------------------------------
    # ICLR 2025
    # Rating scale: discrete {1, 3, 5, 6, 8, 10}  (NOT a continuous 1–10)
    # Decisions public as of early 2026; acceptance rate ~32%.
    # Additional per-review fields: soundness, presentation, contribution.
    # -----------------------------------------------------------------------
    "ICLR_2025": {
        "venue_id": "ICLR.cc/2025/Conference",
        "submission_inv": "ICLR.cc/2025/Conference/-/Submission",
        "review_inv_suffix": "Official_Review",
        "score_field": "rating",          # discrete: 1, 3, 5, 6, 8, 10
        "decision_inv": "ICLR.cc/2025/Conference/-/Decision",
        "decision_field": "decision",
        "extra_score_fields": ["soundness", "presentation", "contribution"],
    },
    # -----------------------------------------------------------------------
    # ICML 2025
    # Conference held July 13–19 2025, Vancouver.
    # Decisions/papers should be public by March 2026.
    # -----------------------------------------------------------------------
    "ICML_2025": {
        "venue_id": "ICML.cc/2025/Conference",
        "submission_inv": "ICML.cc/2025/Conference/-/Submission",
        "review_inv_suffix": "Review",
        "score_field": "rating",
        "decision_inv": "ICML.cc/2025/Conference/-/Decision",
        "decision_field": "decision",
        "extra_score_fields": ["confidence"],
    },
    # -----------------------------------------------------------------------
    # ICLR 2024  (kept as fallback / comparison year)
    # Rating scale: 1–10 continuous with text labels.
    # -----------------------------------------------------------------------
    "ICLR_2024": {
        "venue_id": "ICLR.cc/2024/Conference",
        "submission_inv": "ICLR.cc/2024/Conference/-/Submission",
        "review_inv_suffix": "Official_Review",
        "score_field": "rating",          # e.g. "6: marginally above threshold"
        "decision_inv": "ICLR.cc/2024/Conference/-/Decision",
        "decision_field": "decision",
        "extra_score_fields": [],
    },
    "ICLR_2023": {
        "venue_id": "ICLR.cc/2023/Conference",
        "submission_inv": "ICLR.cc/2023/Conference/-/Submission",
        "review_inv_suffix": "Official_Review",
        "score_field": "rating",
        "decision_inv": "ICLR.cc/2023/Conference/-/Decision",
        "decision_field": "decision",
        "extra_score_fields": [],
    },
    # -----------------------------------------------------------------------
    # NeurIPS 2024
    # Additional fields: correctness, novelty, presentation.
    # -----------------------------------------------------------------------
    "NeurIPS_2024": {
        "venue_id": "NeurIPS.cc/2024/Conference",
        "submission_inv": "NeurIPS.cc/2024/Conference/-/Submission",
        "review_inv_suffix": "Review",
        "score_field": "rating",
        "decision_inv": "NeurIPS.cc/2024/Conference/-/Decision",
        "decision_field": "decision",
        "extra_score_fields": ["correctness", "novelty", "presentation"],
    },
    "ICML_2024": {
        "venue_id": "ICML.cc/2024/Conference",
        "submission_inv": "ICML.cc/2024/Conference/-/Submission",
        "review_inv_suffix": "Review",
        "score_field": "rating",
        "decision_inv": "ICML.cc/2024/Conference/-/Decision",
        "decision_field": "decision",
        "extra_score_fields": [],
    },
}

DATA_DIR = Path(__file__).parent / "data"


def get_client() -> openreview.api.OpenReviewClient:
    """
    Return an OpenReview v2 client.
    If OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD are set (in .env or shell),
    logs in — required for ICLR 2025 and other gated venues.
    Falls back to anonymous access for fully public venues.
    """
    # v2 API login: pass username/password only when both are provided.
    # The v2 endpoint uses 'id' internally — do NOT pass them as keyword
    # args if empty, as that triggers a 400 AdditionalPropertiesError.
    username = os.environ.get("OPENREVIEW_USERNAME", "").strip()
    password = os.environ.get("OPENREVIEW_PASSWORD", "").strip()
    if username and password:
        return openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net",
            username=username,
            password=password,
        )
    return openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")


def _parse_score(raw) -> int | None:
    """
    Normalize a rating field to an integer.
    Handles formats like:
      - 6
      - "6: marginally above threshold"
      - "8: accept"
    """
    if raw is None:
        return None
    m = re.match(r"^\s*(\d+)", str(raw))
    return int(m.group(1)) if m else None


def fetch_reviews_for_paper(
    client: openreview.api.OpenReviewClient,
    forum_id: str,
    score_field: str,
    extra_fields: list[str] | None = None,
) -> tuple[list[int], dict[str, list[int]]]:
    """
    Return (primary_scores, extra_scores) from official reviews for a paper.

    primary_scores: list of ints for score_field across all reviewers.
    extra_scores:   dict mapping extra field name → list of ints (ICLR 2025:
                    soundness, presentation, contribution).
    """
    try:
        notes = client.get_all_notes(forum=forum_id)
    except Exception:
        return [], {}

    primary_scores = []
    extra_scores: dict[str, list[int]] = {f: [] for f in (extra_fields or [])}

    for note in notes:
        # v2 API uses `invitations` (list); v1 used `invitation` (string)
        invs = getattr(note, "invitations", None) or [getattr(note, "invitation", "") or ""]
        inv = " ".join(invs)
        if "Official_Review" not in inv and "Review" not in inv:
            continue

        val = note.content.get(score_field, {})
        if isinstance(val, dict):
            val = val.get("value")
        score = _parse_score(val)
        if score is not None:
            primary_scores.append(score)

        for field in (extra_fields or []):
            ev = note.content.get(field, {})
            if isinstance(ev, dict):
                ev = ev.get("value")
            escore = _parse_score(ev)
            if escore is not None:
                extra_scores[field].append(escore)

    return primary_scores, extra_scores


def fetch_decision(
    client: openreview.api.OpenReviewClient,
    forum_id: str,
    decision_field: str,
) -> str | None:
    """Return the decision string for a paper (Accept / Reject / etc.)."""
    try:
        notes = client.get_all_notes(forum=forum_id)
    except Exception:
        return None

    for note in notes:
        invs = getattr(note, "invitations", None) or [getattr(note, "invitation", "") or ""]
        inv = " ".join(invs)
        if "Decision" not in inv:
            continue
        val = note.content.get(decision_field, {})
        if isinstance(val, dict):
            val = val.get("value")
        if val:
            return str(val)
    return None


def fetch_papers(
    venue_key: str = "ICLR_2025",
    n_papers: int = 50,
    require_reviews: bool = True,
    delay: float = 0.3,
) -> list[dict]:
    """
    Fetch up to n_papers submissions from the given venue, including their
    abstracts and human review scores.

    Args:
        venue_key:       Key into VENUE_CONFIGS.
        n_papers:        Maximum number of papers to return.
        require_reviews: If True, skip papers with no review scores.
        delay:           Seconds to sleep between API calls (rate limiting).

    Returns:
        List of paper dicts.
    """
    cfg = VENUE_CONFIGS[venue_key]
    client = get_client()

    print(f"Fetching submissions from {cfg['venue_id']} ...")
    # get_notes() accepts limit; get_all_notes() paginates everything and does not.
    submissions = client.get_notes(
        invitation=cfg["submission_inv"],
        limit=n_papers * 3,   # overfetch — many may lack reviews
    )
    print(f"  Retrieved {len(submissions)} raw submissions.")

    papers = []
    for sub in tqdm(submissions, desc="Processing papers"):
        c = sub.content

        # Extract fields (v2 API wraps values in {"value": ...})
        def _val(field):
            v = c.get(field, {})
            return v.get("value") if isinstance(v, dict) else v

        title = _val("title") or ""
        abstract = _val("abstract") or ""

        if not title or not abstract:
            continue

        extra_fields = cfg.get("extra_score_fields", [])
        scores, extra_scores = fetch_reviews_for_paper(
            client, sub.id, cfg["score_field"], extra_fields
        )
        if require_reviews and not scores:
            time.sleep(delay)
            continue

        decision = fetch_decision(client, sub.id, cfg["decision_field"])

        paper = {
            "id": sub.id,
            "venue": venue_key,
            "title": title,
            "abstract": abstract,
            "keywords": _val("keywords") or [],
            "human_scores": scores,
            "avg_human_score": round(sum(scores) / len(scores), 2) if scores else None,
            "decision": decision,
        }
        # Store per-field averages for venues with multi-dimensional ratings
        # (e.g. ICLR 2025: soundness, presentation, contribution)
        for field, fscores in extra_scores.items():
            paper[f"avg_{field}"] = round(sum(fscores) / len(fscores), 2) if fscores else None
        papers.append(paper)

        if len(papers) >= n_papers:
            break
        time.sleep(delay)

    return papers


def fetch_balanced_dataset(
    venue_key: str = "ICLR_2025",
    n_per_class: int = 15,
) -> list[dict]:
    """
    Return a balanced dataset with roughly equal accepted and rejected papers.
    Falls back to unbalanced if one class is unavailable.
    """
    papers = fetch_papers(venue_key=venue_key, n_papers=n_per_class * 6, require_reviews=True)

    accepted, rejected = [], []
    for p in papers:
        d = (p.get("decision") or "").lower()
        if "accept" in d:
            accepted.append(p)
        elif "reject" in d:
            rejected.append(p)

    # Balance
    n = min(n_per_class, len(accepted), len(rejected))
    if n == 0:
        print("Warning: could not balance dataset — returning all papers.")
        return papers[: n_per_class * 2]

    balanced = accepted[:n] + rejected[:n]
    print(f"Balanced dataset: {n} accepted + {n} rejected = {len(balanced)} papers.")
    return balanced


def save_papers(papers: list[dict], path: Path | None = None) -> Path:
    """Save papers list to JSON."""
    if path is None:
        path = DATA_DIR / "papers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(papers)} papers to {path}")
    return path


def load_papers(path: Path | None = None) -> list[dict]:
    """Load papers from JSON."""
    if path is None:
        path = DATA_DIR / "papers.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_papers_by_ids(
    paper_ids: list[str],
    score_field: str = "rating",
    extra_fields: list[str] | None = None,
    decision_field: str = "decision",
    delay: float = 0.3,
) -> list[dict]:
    """
    Fetch specific papers by their OpenReview forum IDs.

    Args:
        paper_ids:      List of OpenReview paper/forum IDs.
        score_field:    Review score field name (default: "rating").
        extra_fields:   Additional score fields to extract.
        decision_field: Decision field name (default: "decision").
        delay:          Seconds to sleep between API calls.

    Returns:
        List of paper dicts (same schema as fetch_papers).
    """
    client = get_client()
    papers = []

    for pid in tqdm(paper_ids, desc="Fetching papers by ID"):
        try:
            sub = client.get_note(pid)
        except Exception as e:
            print(f"  Warning: could not fetch {pid}: {e}")
            time.sleep(delay)
            continue

        c = sub.content

        def _val(field):
            v = c.get(field, {})
            return v.get("value") if isinstance(v, dict) else v

        title = _val("title") or ""
        abstract = _val("abstract") or ""

        scores, extra_scores = fetch_reviews_for_paper(
            client, sub.id, score_field, extra_fields or []
        )
        decision = fetch_decision(client, sub.id, decision_field)

        paper = {
            "id": sub.id,
            "venue": _val("venue") or _val("venueid") or "",
            "title": title,
            "abstract": abstract,
            "keywords": _val("keywords") or [],
            "human_scores": scores,
            "avg_human_score": round(sum(scores) / len(scores), 2) if scores else None,
            "decision": decision,
        }
        for field, fscores in extra_scores.items():
            paper[f"avg_{field}"] = round(sum(fscores) / len(fscores), 2) if fscores else None

        papers.append(paper)
        time.sleep(delay)

    return papers


def download_pdfs(
    paper_ids: list[str],
    out_dir: Path | None = None,
    delay: float = 0.5,
) -> list[Path]:
    """
    Download PDFs for given OpenReview paper IDs.
    Requires OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD in .env for gated venues.

    Returns list of saved PDF paths.
    """
    if out_dir is None:
        out_dir = DATA_DIR / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = get_client()
    saved = []

    for pid in tqdm(paper_ids, desc="Downloading PDFs"):
        out_path = out_dir / f"{pid}.pdf"
        if out_path.exists():
            print(f"  Already exists: {out_path}")
            saved.append(out_path)
            continue
        try:
            pdf_bytes = client.get_pdf(pid, is_reference=False)
            out_path.write_bytes(pdf_bytes)
            print(f"  Saved: {out_path}")
            saved.append(out_path)
        except Exception as e:
            print(f"  Warning: could not download {pid}: {e}")
        time.sleep(delay)

    return saved


if __name__ == "__main__":
    ids = ["PwxYoMvmvy", "ONfWFluZBI", "zkNCWtw2fd", "viQ1bLqKY0"]

    # Fetch metadata
    papers = fetch_papers_by_ids(ids)
    if papers:
        save_papers(papers, DATA_DIR / "papers_by_id.json")
        for p in papers:
            print(f"  [{p['id']}] {p['title']} | scores={p['human_scores']} | decision={p['decision']}")

    # Download PDFs
    download_pdfs(ids)
