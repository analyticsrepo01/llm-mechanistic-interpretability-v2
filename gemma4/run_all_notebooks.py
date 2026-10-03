#!/usr/bin/env python3
"""
Run all Gemma 4 notebooks in order and report results.
Usage: ~/miniconda3/envs/llm/bin/python run_all_notebooks.py
"""
import subprocess
import sys
import os
import time
import json

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.expanduser("~/hook_outputs/charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOTEBOOKS = [
    "01_gemma4_hooks_inference.ipynb",
    "02_gemma4_logit_lens.ipynb",
    "03_gemma4_persona_vectors.ipynb",
    "04_gemma4_induction_heads.ipynb",
    "05_gemma4_sparse_autoencoder.ipynb",
    "06_gemma4_attention_composition.ipynb",
]

PYTHON = os.path.expanduser("~/miniconda3/envs/llm/bin/python")
NBCONVERT = [PYTHON, "-m", "jupyter", "nbconvert"]
EXECUTED_DIR = os.path.expanduser("~/gemma4_opening_blackbox/executed")
os.makedirs(EXECUTED_DIR, exist_ok=True)

results = []
total_start = time.time()

print("=" * 70)
print("RUNNING ALL GEMMA 4 NOTEBOOKS")
print("=" * 70)

for nb in NOTEBOOKS:
    nb_path = os.path.join(NOTEBOOK_DIR, nb)
    if not os.path.exists(nb_path):
        print("\n[SKIP] {} — file not found".format(nb))
        results.append({"notebook": nb, "status": "skipped", "reason": "not found"})
        continue

    print("\n" + "=" * 70)
    print("[RUN] {}".format(nb))
    print("=" * 70)

    t0 = time.time()
    try:
        proc = subprocess.run(
            NBCONVERT + [
                "--to", "notebook",
                "--execute",
                "--output-dir", EXECUTED_DIR,
                "--ExecutePreprocessor.timeout=600",
                "--ExecutePreprocessor.kernel_name=python3",
                nb_path,
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        elapsed = time.time() - t0

        if proc.returncode == 0:
            print("[PASS] {} — {:.1f}s".format(nb, elapsed))
            results.append({"notebook": nb, "status": "pass", "time_s": round(elapsed, 1)})
        else:
            print("[FAIL] {} — {:.1f}s".format(nb, elapsed))
            stderr_lines = proc.stderr.strip().split("\n")
            for line in stderr_lines[-30:]:
                print("  {}".format(line))
            results.append({
                "notebook": nb,
                "status": "fail",
                "time_s": round(elapsed, 1),
                "error": "\n".join(stderr_lines[-10:]),
            })
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print("[TIMEOUT] {} — exceeded 900s".format(nb))
        results.append({"notebook": nb, "status": "timeout", "time_s": round(elapsed, 1)})
    except Exception as e:
        elapsed = time.time() - t0
        print("[ERROR] {} — {}".format(nb, str(e)))
        results.append({"notebook": nb, "status": "error", "time_s": round(elapsed, 1), "error": str(e)})

total_time = time.time() - total_start

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
passed = sum(1 for r in results if r["status"] == "pass")
failed = sum(1 for r in results if r["status"] == "fail")
skipped = sum(1 for r in results if r["status"] == "skipped")
print("Total: {} | Passed: {} | Failed: {} | Skipped: {}".format(
    len(results), passed, failed, skipped))
print("Total time: {:.1f}s ({:.1f} min)".format(total_time, total_time / 60))
print()

for r in results:
    icon = {"pass": "OK", "fail": "FAIL", "skipped": "SKIP", "timeout": "TIMEOUT", "error": "ERR"}.get(r["status"], "?")
    time_str = " ({:.1f}s)".format(r.get("time_s", 0)) if "time_s" in r else ""
    print("  [{}] {}{}".format(icon, r["notebook"], time_str))

# Save results
results_path = os.path.join(OUTPUT_DIR, "gemma4_notebook_run_results.json")
with open(results_path, "w") as f:
    json.dump({"results": results, "total_time_s": round(total_time, 1)}, f, indent=2)
print("\nResults saved to: {}".format(results_path))

# List generated Gemma charts
print("\nGenerated Gemma 4 charts:")
if os.path.exists(OUTPUT_DIR):
    charts = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("g4_") and f.endswith(".png")])
    for c in charts:
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, c)) // 1024
        print("  {} ({} KB)".format(c, size_kb))
    print("\nTotal Gemma 4 charts: {}".format(len(charts)))
