#!/usr/bin/env python3
"""
paperreview_automize.py - Automate paper submission to paperreview.ai

Usage:
    python paperreview_automize.py
    python paperreview_automize.py --pdf path/to/paper.pdf --email user@example.com
    python paperreview_automize.py --pdf paper.pdf --email user@example.com --venue ICLR
    python paperreview_automize.py --pdf paper.pdf --email user@example.com --no-headless

Requirements:
    pip install playwright
    playwright install chromium
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    print("[ERROR] playwright not installed.")
    print("  Run: pip install playwright && playwright install chromium")
    sys.exit(1)


SUBMIT_URL = "https://paperreview.ai/"
DEFAULT_LOG_FILE = "paperreview_submissions.json"

VENUES = [
    "ICLR", "NeurIPS", "ICML", "CVPR", "AAAI", "IJCAI",
    "ACL", "EMNLP", "OSDI", "SOSP", "VLDB", "SIGMOD", "Other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate paper submission to paperreview.ai",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="test_pdf.pdf",
        help="Path to the PDF file to submit",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="id.4.github.chjwon@gmail.com",
        help="Email address to receive review notification",
    )
    parser.add_argument(
        "--venue",
        type=str,
        default=None,
        choices=VENUES,
        help="Target venue for the paper review (optional)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help="Path to the JSON log file for saving submission results",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        default=False,
        help="Show browser window (useful for debugging; default: headless)",
    )
    return parser.parse_args()


def load_log(log_file: str) -> list:
    """Load existing submission log entries from JSON file."""
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, list) else []
            except json.JSONDecodeError:
                return []
    return []


def save_log(log_file: str, entries: list) -> None:
    """Append and save all log entries to JSON file."""
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"[LOG] Submission log saved to: {os.path.abspath(log_file)}")


def extract_token(text: str) -> Optional[str]:
    """
    Try to extract a submission token from page text.
    Looks for UUIDs and token-prefixed alphanumeric strings.
    """
    # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    uuid_pattern = r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b'
    uuid_matches = re.findall(uuid_pattern, text, re.IGNORECASE)
    if uuid_matches:
        return uuid_matches[0]

    # "Token: <value>" or "Submission ID: <value>" pattern
    labeled_pattern = r'(?:token|submission[_\s]?id|review[_\s]?id)[:\s]+([A-Za-z0-9_\-]{6,64})'
    labeled_matches = re.findall(labeled_pattern, text, re.IGNORECASE)
    if labeled_matches:
        return labeled_matches[0]

    return None


def submit_paper(
    pdf_path: str,
    email: str,
    venue: Optional[str],
    headless: bool,
) -> dict:
    """
    Submit a paper PDF and email to paperreview.ai via browser automation.
    Returns a dict with submission details including any token displayed on the page.
    """
    pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    result = {
        "timestamp": datetime.now().isoformat(),
        "pdf_path": pdf_path,
        "pdf_name": Path(pdf_path).name,
        "email": email,
        "venue": venue,
        "status": "unknown",
        "token": None,
        "confirmation_text": None,
        "screenshot": None,
        "error": None,
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        try:
            print(f"[INFO] Navigating to {SUBMIT_URL}")
            page.goto(SUBMIT_URL, wait_until="networkidle", timeout=30_000)

            # --- Upload PDF ---
            print(f"[INFO] Uploading PDF: {pdf_path}")
            file_input = page.locator('input[type="file"][accept=".pdf"]')
            file_input.set_input_files(pdf_path)

            # --- Fill email ---
            print(f"[INFO] Entering email: {email}")
            email_input = page.locator('input[type="email"]')
            email_input.fill(email)

            # --- Select venue (optional) ---
            if venue:
                print(f"[INFO] Selecting venue: {venue}")
                venue_select = page.locator("select").first
                all_options = venue_select.locator("option").all_text_contents()
                matched = next((opt for opt in all_options if venue in opt), None)
                if matched:
                    venue_select.select_option(label=matched)
                else:
                    # Fall back to "Other" and type the venue name
                    other_opt = next((opt for opt in all_options if "Other" in opt), None)
                    if other_opt:
                        venue_select.select_option(label=other_opt)
                        # The "Specify Venue" text field should now appear
                        try:
                            page.wait_for_selector('input[placeholder*="venue"]', timeout=5_000)
                            page.locator('input[placeholder*="venue"]').fill(venue)
                        except PlaywrightTimeout:
                            # Try a generic last text input
                            page.locator('input[type="text"]').last.fill(venue)

            # --- Submit form ---
            print("[INFO] Submitting form...")
            submit_btn = page.locator('button[type="submit"]')
            submit_btn.click()

            # --- Step 1: Wait for upload to finish (uploading indicator disappears) ---
            print("[INFO] Waiting for file upload to complete...")
            try:
                page.wait_for_function(
                    """() => {
                        const text = document.body.innerText.toLowerCase();
                        return !text.includes('uploading file');
                    }""",
                    timeout=60_000,
                )
                print("[INFO] Upload complete.")
            except PlaywrightTimeout:
                print("[WARN] Upload indicator still present after 60 s; continuing...")

            # --- Step 2: Wait up to 2 min for token or submission confirmation ---
            print("[INFO] Waiting up to 2 min for token/confirmation to appear...")
            token_found = False
            deadline = time.time() + 120  # 2 minutes
            poll_interval = 5  # check every 5 seconds

            while time.time() < deadline:
                page_text = page.inner_text("body")
                token = extract_token(page_text)
                if token:
                    result["token"] = token
                    result["confirmation_text"] = page_text[:3000]
                    print(f"[INFO] Token found on page: {token}")
                    token_found = True
                    break

                # Also check for explicit success/submitted keywords (outside nav links)
                # Strip nav text (first ~200 chars typically contain nav) before checking
                body_main = page_text[200:].lower()
                if any(kw in body_main for kw in ("submitted", "success", "your submission")):
                    result["confirmation_text"] = page_text[:3000]
                    print("[INFO] Submission confirmed on page (no token visible yet).")
                    token_found = True  # success even without token
                    break

                remaining = int(deadline - time.time())
                print(f"[INFO] Token not yet visible, retrying... ({remaining}s remaining)")
                time.sleep(poll_interval)

            if not token_found:
                # Capture final page state after timeout
                page_text = page.inner_text("body")
                result["confirmation_text"] = page_text[:3000]
                print(f"[INFO] Token not found after 2 min — it will be emailed to {email}")

            # --- Screenshot as proof ---
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"submission_{ts}.png"
            page.screenshot(path=screenshot_path, full_page=True)
            result["screenshot"] = os.path.abspath(screenshot_path)
            print(f"[INFO] Screenshot saved: {screenshot_path}")

            result["status"] = "submitted"
            print("[SUCCESS] Paper submitted successfully!")

        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            print(f"[ERROR] Submission failed: {exc}")
            # Capture error screenshot
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                err_shot = f"submission_error_{ts}.png"
                page.screenshot(path=err_shot, full_page=True)
                result["screenshot"] = os.path.abspath(err_shot)
                print(f"[INFO] Error screenshot saved: {err_shot}")
            except Exception:
                pass

        finally:
            context.close()
            browser.close()

    return result


def print_summary(result: dict) -> None:
    """Print a human-readable submission summary."""
    print()
    print("=" * 60)
    print("SUBMISSION SUMMARY")
    print("=" * 60)
    print(f"Status    : {result['status']}")
    print(f"Timestamp : {result['timestamp']}")
    print(f"PDF       : {result['pdf_name']}")
    print(f"Email     : {result['email']}")
    print(f"Venue     : {result['venue'] or 'Not specified'}")
    if result["token"]:
        print(f"Token     : {result['token']}")
    else:
        print(f"Token     : [Will be emailed to {result['email']} when review is ready]")
    if result["screenshot"]:
        print(f"Screenshot: {result['screenshot']}")
    if result["error"]:
        print(f"Error     : {result['error']}")
    print("=" * 60)


def main() -> int:
    args = parse_args()
    headless = not args.no_headless

    print("=" * 60)
    print("paperreview.ai Automated Submission Tool")
    print("=" * 60)
    print(f"PDF       : {args.pdf}")
    print(f"Email     : {args.email}")
    print(f"Venue     : {args.venue or 'Not specified'}")
    print(f"Headless  : {headless}")
    print(f"Log file  : {args.log_file}")
    print("=" * 60)

    # Submit paper
    try:
        result = submit_paper(args.pdf, args.email, args.venue, headless)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    # Persist to log
    entries = load_log(args.log_file)
    entries.append(result)
    save_log(args.log_file, entries)

    print_summary(result)
    return 0 if result["status"] == "submitted" else 1


if __name__ == "__main__":
    sys.exit(main())
