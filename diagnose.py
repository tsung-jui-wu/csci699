"""
diagnose.py

Run this to debug OpenReview connectivity before running the main pipeline.
Checks credentials, lists available invitations, and tries candidate
invitation strings for ICLR 2025.

Usage:
    python diagnose.py
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import openreview

VENUE_ID = "ICLR.cc/2025/Conference"

# Candidate submission invitation strings to try
CANDIDATES = [
    f"{VENUE_ID}/-/Blind_Submission",
    f"{VENUE_ID}/-/Submission",
    f"{VENUE_ID}/-/Draft_Submission",
    f"{VENUE_ID}/-/Camera_Ready_Submission",
]


def get_client():
    username = os.environ.get("OPENREVIEW_USERNAME", "")
    password = os.environ.get("OPENREVIEW_PASSWORD", "")
    return openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=username or None,
        password=password or None,
    )


def check_credentials(client):
    print("── Credential check ─────────────────────────────────")
    username = os.environ.get("OPENREVIEW_USERNAME", "")
    if not username:
        print("  WARNING: OPENREVIEW_USERNAME not set in .env — using anonymous access.")
        print("  Anonymous access is blocked for ICLR 2025 submissions.")
        return
    try:
        profile = client.get_profile(username)
        print(f"  Logged in as: {profile.id}")
    except Exception as e:
        print(f"  Login failed: {e}")
        print("  Check your OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD in .env")


def list_invitations(client):
    print("\n── Available invitations for the venue ─────────────")
    try:
        invitations = client.get_all_invitations(replyto=VENUE_ID)
        if not invitations:
            invitations = client.get_all_invitations(prefix=VENUE_ID)
        if invitations:
            for inv in invitations[:20]:
                print(f"  {inv.id}")
        else:
            print("  None found — venue may not be accessible with current credentials.")
    except Exception as e:
        print(f"  Could not list invitations: {e}")


def try_candidates(client):
    print("\n── Testing candidate invitation strings ─────────────")
    for inv_str in CANDIDATES:
        try:
            notes = client.get_notes(invitation=inv_str, limit=1)
            count = len(notes)
            status = f"✓  {count} result(s)" if count else "✗  0 results"
        except Exception as e:
            status = f"✗  Error: {e}"
        print(f"  {inv_str}")
        print(f"     {status}")


def try_venueid_query(client):
    """Newer API: query by venueid content field instead of invitation."""
    print("\n── Trying venueid content-field query ───────────────")
    try:
        notes = client.get_notes(
            content={"venueid": VENUE_ID},
            limit=3,
        )
        if notes:
            print(f"  ✓ Found {len(notes)} note(s) via venueid filter.")
            print(f"    Sample title: {notes[0].content.get('title', {}).get('value', 'N/A')}")
            print(f"    Sample invitation: {notes[0].invitation}")
        else:
            print("  ✗ 0 results via venueid filter.")
    except Exception as e:
        print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    print(f"Diagnosing OpenReview access for: {VENUE_ID}\n")
    client = get_client()
    check_credentials(client)
    list_invitations(client)
    try_candidates(client)
    try_venueid_query(client)
    print("\nDone. Use the working invitation string in VENUE_CONFIGS['ICLR_2025']['submission_inv'].")
