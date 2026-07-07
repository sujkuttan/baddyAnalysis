"""Colab-ready entry. Run cells in order, or `python notebooks/run_colab.py --video ...`."""

import os
import json

from src import pipeline, stabilize


def select_corners(video_path, out_json="corners.json"):
    corners = stabilize.interactive_select_corners(video_path)
    json.dump({"corners": [[int(x), int(y)] for x, y in corners]}, open(out_json, "w"))
    print("saved corners to", out_json)
    return corners


def run(video_path, corners_json, labels_csv="labels_import.csv", device="cuda"):
    corners = json.load(open(corners_json))["corners"]
    res = pipeline.run_full_pipeline(
        video_path, corners, out_dir="data", labels_csv=labels_csv,
        device=device, tracknet_weights="weights/TrackNet_best.pt",
    )
    print("predictions:", res["predictions_csv"])
    print("metrics:", res["metrics"])
    return res


def compare(labels_csv="labels_import.csv", new="data/new_predictions.csv", bst=None):
    return pipeline.ab_compare(labels_csv, new, bst)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--corners", required=True)
    ap.add_argument("--labels", default="labels_import.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bst", default=None)
    a = ap.parse_args()
    run(a.video, a.corners, a.labels, a.device)
    compare(a.labels, "data/new_predictions.csv", a.bst)
