"""Drive the real Streamlit app with Playwright and save README screenshots.

Usage:  python scripts/capture_screenshots.py [--out docs/img] [--port 8599]
Requires: pip install playwright && playwright install chromium
Everything shown is produced by the app itself on its built-in synthetic sample
dataset (no real customer data).
"""
import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def goto(page, label):
    page.get_by_test_id("stSidebar").get_by_role("combobox").click()
    page.get_by_role("option", name=label).click()
    page.wait_for_timeout(1200)


def click(page, name, wait=4000):
    page.get_by_role("button", name=name).click()
    page.wait_for_timeout(wait)


def settle(page):
    page.wait_for_selector("[data-testid=stStatusWidget]", state="detached", timeout=180000)
    page.wait_for_timeout(1500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "img")
    ap.add_argument("--port", type=int, default=8599)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/app.py", "--server.headless", "true",
         "--server.port", str(args.port), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://localhost:{args.port}"
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except Exception:
                time.sleep(1)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not args.headed)
            # tall viewport: Streamlit scrolls inside its own container, so full_page would clip
            page = browser.new_page(viewport={"width": 1400, "height": 1500})
            page.goto(url)
            page.wait_for_timeout(3000)

            goto(page, "Data Upload & Processing")
            click(page, "Use Sample Dataset", 3000)
            click(page, "Process Transaction Data", 3000)
            settle(page)

            goto(page, "RFM Analysis")
            click(page, "Run RFM Analysis")
            settle(page)
            page.screenshot(path=args.out / "rfm.png")

            goto(page, "Churn Prediction")
            click(page, "Train Churn Models", 1000)
            settle(page)
            page.screenshot(path=args.out / "churn.png")
            click(page, "Predict Churn Risk", 1000)
            settle(page)

            goto(page, "Customer Lifetime Value")
            click(page, "Calculate Customer Lifetime Value", 1000)
            settle(page)
            page.screenshot(path=args.out / "clv.png")

            goto(page, "Dashboard Overview")
            settle(page)
            page.screenshot(path=args.out / "dashboard.png")
            browser.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
