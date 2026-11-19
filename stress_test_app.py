"""Run the app headless and submit a patient many times in a row.

Catches the Qiskit-inside-Streamlit segfault (Qiskit 2.5+) and any error a
prediction raises. Uses Streamlit's AppTest, so no browser or server is needed.

Usage:  python stress_test_app.py [--runs 15]
Exit code: 0 = all runs passed, 1 = an error. A segfault kills the process
(exit code 139 / -11), which also counts as a failure.
"""
import argparse
import faulthandler
import os
import sys
import warnings

from streamlit.testing.v1 import AppTest

faulthandler.enable()
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=15)
    a = ap.parse_args()
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    at = AppTest.from_file(app, default_timeout=300)
    at.run()
    if at.exception:
        print("App failed on first load:", at.exception)
        return 1
    for i in range(a.runs):
        button = [b for b in at.button if "Check risk" in str(b.label)][0]
        button.click()
        at.run()
        if at.exception:
            print(f"Run {i + 1}: app raised", at.exception)
            return 1
        print(f"run {i + 1}/{a.runs} ok", flush=True)
    print("STRESS TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
