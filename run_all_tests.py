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

    # Run with pytest
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--html", report_file,
        "--self-contained-html",
        "--tb=short",
    ]

    result = subprocess.run(cmd, capture_output=False)
    print(f"\n{'=' * 60}")
    print(f"  Report saved to: {report_file}")
    print(f"  Exit code: {result.returncode}")
    print(f"{'=' * 60}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())