"""Capture dashboard and API-docs screenshots for the README.

Requires the API (port 8000) and dashboard (port 8501) to be running, e.g.
``docker compose up`` or ``make api`` + ``make dashboard``, and Playwright:

    pip install playwright && playwright install chromium
    python scripts/capture_screenshots.py

Writes PNGs to reports/screenshots/.
"""

from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"
DASHBOARD = "http://localhost:8501"
API_DOCS = "http://localhost:8000/docs"

PAGES = [
    ("Overview", "dashboard_overview.png"),
    ("Score a transaction", "dashboard_single.png"),
    ("Batch scoring", "dashboard_batch.png"),
    ("Model performance", "dashboard_performance.png"),
    ("Prediction history", "dashboard_history.png"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page.goto(API_DOCS, wait_until="networkidle")
        page.screenshot(path=OUT / "api_docs.png", full_page=True)
        print("saved api_docs.png")

        page.goto(DASHBOARD, wait_until="networkidle")
        time.sleep(3)
        for label, filename in PAGES:
            page.get_by_text(label, exact=True).first.click()
            time.sleep(3)
            if label == "Score a transaction":
                page.get_by_role("button", name="Score transaction").click()
                time.sleep(3)
            page.screenshot(path=OUT / filename, full_page=True)
            print("saved", filename)
        browser.close()


if __name__ == "__main__":
    main()
