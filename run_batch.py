"""
Batch runner for AgentGuard-Clinical.

Runs the full 5-node pipeline on multiple images and logs results
to a CSV — the start of building your headline finding (how often
does the audit layer flag inconsistency, and on which cases).

Usage:
    python run_batch.py <folder_of_images>

Processes every .jpg/.png directly inside that folder (not recursive).
"""

import os
import sys
import csv
from graph import build_graph


def find_images(folder: str) -> list:
    exts = (".jpg", ".jpeg", ".png")
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(exts)
    ]


def load_already_processed(output_csv: str) -> set:
    """Returns the set of filenames already in the CSV, so a multi-day
    batch run (paced by API quota) doesn't waste calls re-processing
    images from a previous day."""
    if not os.path.exists(output_csv):
        return set()

    processed = set()
    with open(output_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["filename"])
    return processed


def run_batch(folder: str, output_csv: str = "batch_results.csv"):
    all_images = find_images(folder)
    if not all_images:
        print(f"No images found in {folder}")
        return

    already_processed = load_already_processed(output_csv)
    images = [p for p in all_images if os.path.basename(p) not in already_processed]

    if already_processed:
        print(f"Skipping {len(already_processed)} already-processed images from a previous run.")

    if not images:
        print("Nothing new to process — all images in this folder are already in the CSV.")
        return

    app = build_graph()
    new_rows = []

    for i, image_path in enumerate(images, 1):
        print(f"\n--- [{i}/{len(images)}] {os.path.basename(image_path)} ---")

        initial_state = {
            "image_path": image_path,
            "diagnosis": None,
            "confidence": None,
            "logits": None,
            "class_names": None,
            "reasoning_text": None,
            "reasoning_is_fallback": None,
            "heatmap_path": None,
            "heatmap_array": None,
            "region_label": None,
            "audit_verdict": None,
            "audit_explanation": None,
            "needs_human_review": None,
            "review_reason": None,
            "error": None,
        }

        try:
            final_state = app.invoke(initial_state)
        except Exception as e:
            print(f"FAILED: {e}")
            new_rows.append({
                "filename": os.path.basename(image_path),
                "diagnosis": "ERROR",
                "confidence": "",
                "audit_verdict": "",
                "error": str(e),
            })
            continue

        new_rows.append({
            "filename": os.path.basename(image_path),
            "diagnosis": final_state.get("diagnosis"),
            "confidence": final_state.get("confidence"),
            "audit_verdict": final_state.get("audit_verdict"),
            "error": final_state.get("error"),
        })

    # Append new rows to the CSV (write header only if the file is new)
    file_exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "diagnosis", "confidence", "audit_verdict", "error"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    # Print summary — for ALL rows in the file so far, not just this run
    all_processed = already_processed | {r["filename"] for r in new_rows}
    with open(output_csv, "r", newline="") as f:
        all_rows = list(csv.DictReader(f))

    consistent = sum(1 for r in all_rows if r["audit_verdict"] == "consistent")
    inconsistent = sum(1 for r in all_rows if r["audit_verdict"] == "inconsistent")
    uncertain = sum(1 for r in all_rows if r["audit_verdict"] == "uncertain")
    errors = sum(1 for r in all_rows if r["diagnosis"] == "ERROR")

    print(f"\n=== THIS RUN: {len(new_rows)} new images processed ===")
    print(f"=== CUMULATIVE TOTAL: {len(all_rows)} images across all runs ===")
    print(f"Consistent:   {consistent}")
    print(f"Inconsistent: {inconsistent}")
    print(f"Uncertain:    {uncertain}")
    print(f"Errors:       {errors}")
    print(f"\nFull results saved to: {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_batch.py <folder_of_images>")
        sys.exit(1)

    run_batch(sys.argv[1])