#!/usr/bin/env python3
"""Run all AI test project tests and generate reports."""
import sys
import os
import subprocess
from datetime import datetime

# Ensure we're in the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

REPORT_DIR = "reports"

def run_tests():
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORT_DIR, f"ai_test_report_{timestamp}.html")

    print("=" * 60)
    print(f"  AI Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check if pytest-html is available
    try:
        import pytest_html  # noqa: F401
        has_html = True
    except ImportError:
        has_html = False

    # Build command
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
    ]
    if has_html:
        cmd += ["--html", report_file, "--self-contained-html"]

    result = subprocess.run(cmd, capture_output=False)

    if has_html:
        print(f"\n{'=' * 60}")
        print(f"  HTML report: {report_file}")
    else:
        print(f"\n{'=' * 60}")
        print(f"  Tip: pip install pytest-html  →  get HTML report")
    print(f"  Exit code: {result.returncode}")
    print(f"{'=' * 60}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())