"""
run_reviews.py

Automates uploading PDFs to paperreview.ai using Playwright and scrapes
the resulting review text.

paperreview.ai workflow (as of 2025):
  1. Navigate to https://paperreview.ai
  2. Click the file upload area or "Upload PDF" button
  3. Wait for processing (typically 30–90 seconds — it fetches related work)
  4. Scrape the review text from the results panel

Run `playwright install chromium` once before using this module.

NOTE: This module respects the site's terms of service — only for research
on copies of publicly available papers, never for gaming real submissions.
"""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Optional

REVIEWS_DIR = Path(__file__).parent / "data" / "reviews"
PAPERREVIEW_URL = "https://paperreview.ai"

# Selector hints — inspect actual DOM if these break after site updates
_UPLOAD_INPUT_SELECTOR = "input[type='file']"
_RESULT_CONTAINER_SELECTORS = [
    "[class*='review']",
    "[class*='result']",
    "[class*='output']",
    "article",
    "main",
]
_PROCESSING_INDICATORS = ["processing", "loading", "analyzing", "fetching"]
_MAX_WAIT_SECONDS = 180


async def _wait_for_review(page, timeout: int = _MAX_WAIT_SECONDS) -> str:
    """
    Poll the page until the review text appears (no spinner / loading text).
    Returns the scraped review text.
    """
    start = time.time()
    while time.time() - start < timeout:
        # Check if still loading
        body_text = await page.inner_text("body")
        lower = body_text.lower()
        still_loading = any(ind in lower for ind in _PROCESSING_INDICATORS)

        # Look for substantial review content (heuristic: > 300 chars after upload)
        if not still_loading and len(body_text) > 500:
            return body_text

        await asyncio.sleep(3)

    raise TimeoutError(f"paperreview.ai did not return a review within {timeout}s")


async def _scrape_review_sections(page) -> dict[str, str]:
    """
    Attempt to extract structured review sections from the page.
    Returns a dict mapping section name → text.
    Falls back to {'full_text': <entire body>} if structure is unclear.
    """
    # Try known section headings from paperreview.ai's 7-dimension format
    dimensions = [
        "Originality",
        "Importance",
        "Well-supported claims",
        "Soundness",
        "Clarity",
        "Value",
        "Contextualization",
        "Summary",
        "Strengths",
        "Weaknesses",
        "Recommendation",
    ]

    full_text = await page.inner_text("body")
    sections: dict[str, str] = {}

    for dim in dimensions:
        # Regex: from heading to next heading or end of string
        pattern = rf"{re.escape(dim)}[:\s]*(.*?)(?={'|'.join(dimensions)}|$)"
        m = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if m:
            sections[dim] = m.group(1).strip()[:1000]  # cap at 1000 chars

    if not sections:
        sections["full_text"] = full_text

    return sections


async def review_pdf_async(
    pdf_path: Path,
    headless: bool = True,
    save_dir: Path | None = None,
) -> dict:
    """
    Upload a single PDF to paperreview.ai and return the scraped review.

    Args:
        pdf_path:  Path to the PDF file.
        headless:  Run browser in headless mode.
        save_dir:  If given, save raw JSON review to this directory.

    Returns:
        Dict with keys: pdf_path, sections, raw_text, timestamp
    """
    from playwright.async_api import async_playwright

    result = {"pdf_path": str(pdf_path), "sections": {}, "raw_text": "", "error": None}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        try:
            await page.goto(PAPERREVIEW_URL, timeout=30_000)
            await page.wait_for_load_state("networkidle", timeout=15_000)

            # Locate and interact with file upload
            upload_input = page.locator(_UPLOAD_INPUT_SELECTOR)
            await upload_input.wait_for(state="attached", timeout=10_000)
            await upload_input.set_input_files(str(pdf_path))

            # Wait for review to complete
            await _wait_for_review(page, timeout=_MAX_WAIT_SECONDS)

            # Scrape
            result["sections"] = await _scrape_review_sections(page)
            result["raw_text"] = await page.inner_text("body")

        except Exception as e:
            result["error"] = str(e)
        finally:
            await browser.close()

    # Persist raw result
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = pdf_path.stem
        out = save_dir / f"{stem}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def review_pdf(
    pdf_path: Path,
    headless: bool = True,
    save_dir: Path | None = None,
) -> dict:
    """Synchronous wrapper around review_pdf_async."""
    return asyncio.run(review_pdf_async(pdf_path, headless=headless, save_dir=save_dir))


def review_all_pdfs(
    pdf_map: dict[str, dict[str, dict[str, Path]]],
    n_runs: int = 3,
    headless: bool = True,
    save_dir: Path | None = None,
    skip_existing: bool = True,
) -> dict[str, dict[str, dict[str, list[dict]]]]:
    """
    Review all PDFs in pdf_map, running each n_runs times for stability.

    pdf_map layout:  pdf_map[paper_id][strategy][position] = Path

    Returns nested dict with the same structure but values are lists of
    n_runs review dicts.
    """
    if save_dir is None:
        save_dir = REVIEWS_DIR

    results: dict[str, dict[str, dict[str, list[dict]]]] = {}

    for paper_id, strat_dict in pdf_map.items():
        results[paper_id] = {}
        for strategy, pos_dict in strat_dict.items():
            results[paper_id][strategy] = {}
            for position, pdf_path in pos_dict.items():
                runs = []
                for run_idx in range(n_runs):
                    cache_key = f"{pdf_path.stem}__run{run_idx}"
                    cached = save_dir / f"{cache_key}.json"

                    if skip_existing and cached.exists():
                        with open(cached, encoding="utf-8") as f:
                            runs.append(json.load(f))
                        print(f"  [cache] {cache_key}")
                        continue

                    print(f"  Reviewing {pdf_path.name} (run {run_idx + 1}/{n_runs}) ...")
                    review = review_pdf(pdf_path, headless=headless, save_dir=None)

                    # Save with run-specific name
                    with open(cached, "w", encoding="utf-8") as f:
                        json.dump(review, f, indent=2, ensure_ascii=False)

                    runs.append(review)
                    time.sleep(5)  # polite delay between runs

                results[paper_id][strategy][position] = runs

    return results


# ---------------------------------------------------------------------------
# Quick manual test (not headless so you can watch)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_reviews.py <path_to_pdf>")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    print(f"Uploading {pdf} to paperreview.ai ...")
    result = review_pdf(pdf, headless=False, save_dir=REVIEWS_DIR)

    if result["error"]:
        print(f"ERROR: {result['error']}")
    else:
        print("Sections found:")
        for k, v in result["sections"].items():
            print(f"  [{k}]: {v[:120]}...")
