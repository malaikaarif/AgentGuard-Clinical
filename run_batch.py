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


def run_batch(folder: str, output_csv: str = "batch_results.csv"):
    images = find_images(folder)
    if not images:
        print(f"No images found in {folder}")
        return

    app = build_graph()
    rows = []

    for i, image_path in enumerate(images, 1):
        print(f"\n--- [{i}/{len(images)}] {os.path.basename(image_path)} ---")

        initial_state = {
            "image_path": image_path,
            "diagnosis": None,
            "confidence": None,
            "logits": None,
            "class_names": None,
            "reasoning_text": None,
            "heatmap_path": None,
            "heatmap_array": None,
            "region_label": None,
            "audit_verdict": None,
            "audit_explanation": None,
            "error": None,
        }

        try:
            final_state = app.invoke(initial_state)
        except Exception as e:
            print(f"FAILED: {e}")
            rows.append({
                "filename": os.path.basename(image_path),
                "diagnosis": "ERROR",
                "confidence": "",
                "audit_verdict": "",
                "error": str(e),
            })
            continue

        rows.append({
            "filename": os.path.basename(image_path),
            "diagnosis": final_state.get("diagnosis"),
            "confidence": final_state.get("confidence"),
            "audit_verdict": final_state.get("audit_verdict"),
            "error": final_state.get("error"),
        })

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "diagnosis", "confidence", "audit_verdict", "error"])
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    consistent = sum(1 for r in rows if r["audit_verdict"] == "consistent")
    inconsistent = sum(1 for r in rows if r["audit_verdict"] == "inconsistent")
    uncertain = sum(1 for r in rows if r["audit_verdict"] == "uncertain")
    errors = sum(1 for r in rows if r["diagnosis"] == "ERROR")

    print(f"\n=== BATCH SUMMARY ({len(rows)} images) ===")
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