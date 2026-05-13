"""
End-to-end Playwright tests for all 5 SmartCity FSA models.
Run with:  python webapp/test_e2e.py
"""
import time
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright, expect, Page

BASE_URL = "http://localhost:8501"
ROOT_DIR = Path(__file__).resolve().parents[1]

RESULTS: list[tuple[str, bool, str]] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def pass_test(name: str) -> None:
    RESULTS.append((name, True, ""))
    log(f"  ✅ PASS  {name}")


def fail_test(name: str, reason: str) -> None:
    RESULTS.append((name, False, reason))
    log(f"  ❌ FAIL  {name}: {reason}")


def wait_for_streamlit(page: Page, timeout: int = 60_000) -> None:
    """Wait until Streamlit finishes rendering (spinner gone)."""
    page.wait_for_load_state("networkidle", timeout=timeout)
    try:
        page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=10_000)
    except Exception:
        pass
    time.sleep(1.5)


def select_model(page: Page, model_label: str) -> None:
    """Select a model from the Streamlit sidebar custom selectbox."""
    sb = page.locator('[data-testid="stSidebar"] [data-baseweb="select"]').first
    sb.click()
    # Options appear in a popover listbox
    option = page.locator('[data-testid="stSelectboxVirtualDropdown"] li', has_text=model_label).first
    option.wait_for(state="visible", timeout=10_000)
    option.click()
    wait_for_streamlit(page)


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
def test_home(page: Page) -> None:
    log("\n▶  Home page")
    try:
        select_model(page, "Home")
        welcome = page.locator("text=Welcome to SmartCity FSA")
        expect(welcome).to_be_visible(timeout=15_000)
        pass_test("Home page loads")
    except Exception as exc:
        fail_test("Home page", str(exc))


# ---------------------------------------------------------------------------
# Model 1 — Traditional ML
# ---------------------------------------------------------------------------
def test_model1(page: Page) -> None:
    log("\n▶  Model 1: Traditional ML")
    try:
        select_model(page, "Model 1: Traditional ML")

        header = page.locator("h1, h2, h3", has_text="Model 1")
        expect(header.first).to_be_visible(timeout=20_000)
        pass_test("Model 1 page loads")

        # Feature importance chart renders (bar chart at bottom)
        bar_chart = page.locator('[data-testid="stVegaLiteChart"]').first
        expect(bar_chart).to_be_visible(timeout=30_000)
        pass_test("Model 1 feature importance chart visible")

        # Click Predict with default slider/selectbox values
        predict_btn = page.locator('button', has_text="Predict").first
        predict_btn.click()
        wait_for_streamlit(page, timeout=60_000)

        result = page.locator('[data-testid="stAlert"], [data-testid="stAlertContentSuccess"]').first
        expect(result).to_be_visible(timeout=30_000)
        result_text = result.text_content()
        log(f"     Result: {result_text.strip()[:80]}")
        pass_test("Model 1 predict returns a result")

    except Exception as exc:
        fail_test("Model 1", str(exc))


# ---------------------------------------------------------------------------
# Model 2 — Deep Learning
# ---------------------------------------------------------------------------
def test_model2(page: Page) -> None:
    log("\n▶  Model 2: Deep Learning")
    try:
        select_model(page, "Model 2: Deep Learning")

        header = page.locator("h1, h2, h3", has_text="Model 2")
        expect(header.first).to_be_visible(timeout=20_000)
        pass_test("Model 2 page loads")

        # Architecture comparison table
        table = page.locator('[data-testid="stTable"]').first
        expect(table).to_be_visible(timeout=20_000)
        pass_test("Model 2 architecture comparison table visible")

        # Predict
        predict_btn = page.locator('button', has_text="Predict").first
        predict_btn.click()
        wait_for_streamlit(page, timeout=60_000)

        result = page.locator('[data-testid="stAlert"], [data-testid="stAlertContentSuccess"]').first
        expect(result).to_be_visible(timeout=30_000)
        result_text = result.text_content()
        log(f"     Result: {result_text.strip()[:80]}")
        pass_test("Model 2 predict returns a result")

    except Exception as exc:
        fail_test("Model 2", str(exc))


# ---------------------------------------------------------------------------
# Model 3 — CNN (Image Classification)
# ---------------------------------------------------------------------------
def test_model3(page: Page) -> None:
    log("\n▶  Model 3: CNN (Image Classification)")
    try:
        select_model(page, "Model 3: CNN (Image Classification)")

        header = page.locator("h1, h2, h3", has_text="Model 3")
        expect(header.first).to_be_visible(timeout=20_000)
        pass_test("Model 3 page loads")

        # Click "Positive 1" sample button
        sample_btn = page.locator('button', has_text="Positive 1").first
        sample_btn.click()
        wait_for_streamlit(page)

        # Sample image should appear
        img = page.locator('[data-testid="stImage"] img').first
        expect(img).to_be_visible(timeout=15_000)
        pass_test("Model 3 sample image loads")

        # Click Classify and wait for result (CNN inference can be slow)
        classify_btn = page.locator('button', has_text="Classify").first
        classify_btn.click()

        # Poll for result — wait up to 3 minutes for first-run compilation
        result_text = None
        for _ in range(90):
            time.sleep(2)
            for tid in ("stAlertContentSuccess", "stAlertContentError"):
                el = page.locator(f'[data-testid="{tid}"]')
                if el.count() > 0:
                    txt = el.first.text_content()
                    if any(k in txt for k in ("pothole", "Pothole", "confidence")):
                        result_text = txt
                        break
            if result_text:
                break
        if result_text:
            log(f"     Result: {result_text.strip()[:80]}")
            pass_test("Model 3 classification returns a result")
        else:
            fail_test("Model 3 classification", "No result after 3 minutes")

    except Exception as exc:
        fail_test("Model 3", str(exc))


# ---------------------------------------------------------------------------
# Model 4 — NLP (Text Classification)
# ---------------------------------------------------------------------------
def test_model4(page: Page) -> None:
    log("\n▶  Model 4: NLP (Text Classification)")
    try:
        select_model(page, "Model 4: NLP (Text Classification)")

        header = page.locator("h1, h2, h3", has_text="Model 4")
        expect(header.first).to_be_visible(timeout=20_000)
        pass_test("Model 4 page loads")

        # Load sample via "Blocked Driveway" button
        sample_btn = page.locator('button', has_text="Blocked Driveway").first
        sample_btn.click()
        wait_for_streamlit(page)

        # Text area should now have content
        textarea = page.locator('textarea').first
        expect(textarea).not_to_be_empty(timeout=10_000)
        pass_test("Model 4 sample text loaded into textarea")

        # Click Classify
        classify_btn = page.locator('button', has_text="Classify").first
        classify_btn.click()
        wait_for_streamlit(page, timeout=60_000)

        result = page.locator('[data-testid="stAlertContentSuccess"], [data-testid="stAlert"]').first
        expect(result).to_be_visible(timeout=30_000)
        result_text = result.text_content()
        log(f"     Result: {result_text.strip()[:80]}")
        pass_test("Model 4 classification returns a result")

    except Exception as exc:
        fail_test("Model 4", str(exc))


# ---------------------------------------------------------------------------
# Model 5 — Innovation (XGBoost Road Deterioration)
# ---------------------------------------------------------------------------
def test_model5(page: Page) -> None:
    log("\n▶  Model 5: Innovation (Road Deterioration)")
    try:
        select_model(page, "Model 5: Innovation")

        header = page.locator("h1, h2, h3", has_text="Model 5")
        expect(header.first).to_be_visible(timeout=20_000)
        pass_test("Model 5 page loads")

        # Load "Critical" example preset
        critical_btn = page.locator('button', has_text="Critical").first
        critical_btn.click()
        wait_for_streamlit(page)
        pass_test("Model 5 Critical example loaded")

        # Submit the form
        submit_btn = page.locator('button', has_text="Predict Deterioration Level").first
        submit_btn.click()
        wait_for_streamlit(page, timeout=30_000)

        result = page.locator('[data-testid="stAlertContentSuccess"], [data-testid="stAlert"]').first
        expect(result).to_be_visible(timeout=20_000)
        result_text = result.text_content()
        log(f"     Critical result: {result_text.strip()[:80]}")
        pass_test("Model 5 Critical predict returns a result")

        # Also test Low preset
        low_btn = page.locator('button', has_text="Low").first
        low_btn.click()
        wait_for_streamlit(page)
        submit_btn2 = page.locator('button', has_text="Predict Deterioration Level").first
        submit_btn2.click()
        wait_for_streamlit(page, timeout=30_000)

        result2 = page.locator('[data-testid="stAlertContentSuccess"], [data-testid="stAlert"]').first
        expect(result2).to_be_visible(timeout=20_000)
        result2_text = result2.text_content()
        log(f"     Low result:      {result2_text.strip()[:80]}")
        pass_test("Model 5 Low preset predict returns a result")

    except Exception as exc:
        fail_test("Model 5", str(exc))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main() -> int:
    log(f"\n{'='*60}")
    log("  SmartCity FSA — End-to-End Model Tests")
    log(f"  Target: {BASE_URL}")
    log(f"{'='*60}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, slow_mo=200)
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()

        log(f"\nNavigating to {BASE_URL} …")
        page.goto(BASE_URL, timeout=30_000)
        wait_for_streamlit(page, timeout=90_000)
        log("App is live.\n")

        test_home(page)
        test_model1(page)
        test_model2(page)
        test_model3(page)
        test_model4(page)
        test_model5(page)

        browser.close()

    # Summary
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    log(f"\n{'='*60}")
    log(f"  Results: {passed} passed, {failed} failed  (total {len(RESULTS)})")
    log(f"{'='*60}")

    if failed:
        log("\nFailed tests:")
        for name, ok, reason in RESULTS:
            if not ok:
                log(f"  • {name}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
