import csv
import os
import numpy as np

from .config import CANONICAL_STROKES, STROKE_TO_ID, canonical_stroke


def load_labels(path):
    rows = list(csv.DictReader(open(path)))
    return [r for r in rows if r.get("label_status") == "labeled"]


def load_preds(path, key="frame"):
    preds = {}
    if not path or not os.path.exists(path):
        return preds
    for r in csv.DictReader(open(path)):
        k = r.get(key)
        if k is None or str(k).strip() == "":
            continue
        preds[str(int(float(k)))] = r.get("predicted_stroke")
    return preds


def evaluate(gt_list, preds, key="frame"):
    correct = 0
    total = 0
    conf = np.zeros((len(CANONICAL_STROKES), len(CANONICAL_STROKES)), dtype=int)
    per_class = {s: {"tp": 0, "tot": 0} for s in CANONICAL_STROKES}
    for g in gt_list:
        kv = g.get(key)
        if kv is None or str(kv).strip() == "":
            continue
        pk = str(int(float(kv)))
        if pk not in preds:
            continue
        pred_raw = preds[pk]
        pred = canonical_stroke(pred_raw)
        true = canonical_stroke(g["true_stroke"])
        if pred is None or true is None:
            continue
        total += 1
        per_class[true]["tot"] += 1
        if pred == true:
            correct += 1
            per_class[true]["tp"] += 1
        conf[STROKE_TO_ID[true], STROKE_TO_ID[pred]] += 1
    acc = correct / total if total else 0.0
    return {"accuracy": acc, "total": total, "confusion": conf, "per_class": per_class}


def _fmt_confusion(conf):
    head = "        " + " ".join(f"{s[:5]:>5}" for s in CANONICAL_STROKES)
    lines = [head]
    for i, s in enumerate(CANONICAL_STROKES):
        lines.append(f"{s[:6]:>6} " + " ".join(f"{conf[i,j]:>5}" for j in range(len(s))))
    return "\n".join(lines)


def run_ab(labels_csv, new_preds_csv, bst_preds_csv=None, out_csv="data/ab_report.csv"):
    gt = load_labels(labels_csv)
    bst = load_preds(bst_preds_csv, "frame") if bst_preds_csv else {str(int(float(g["frame"]))): g["predicted_stroke"] for g in gt if str(g.get("frame", "")).strip() != ""}
    new = load_preds(new_preds_csv, "frame")

    res_bst = evaluate(gt, bst)
    res_new = evaluate(gt, new)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "n_evaluated"])
        w.writerow(["BST", res_bst["accuracy"], res_bst["total"]])
        w.writerow(["NEW_PIPELINE", res_new["accuracy"], res_new["total"]])

    print("=== A/B on", len(gt), "labeled shots ===")
    print(f"BST          accuracy = {res_bst['accuracy']:.3f}  (n={res_bst['total']})")
    print(_fmt_confusion(res_bst["confusion"]))
    print(f"NEW_PIPELINE accuracy = {res_new['accuracy']:.3f}  (n={res_new['total']})")
    print(_fmt_confusion(res_new["confusion"]))
    print(f"Report written to {out_csv}")
    return {"bst": res_bst, "new": res_new}


if __name__ == "__main__":
    import sys
    labels = sys.argv[1] if len(sys.argv) > 1 else "labels_import.csv"
    newp = sys.argv[2] if len(sys.argv) > 2 else "data/new_predictions.csv"
    bstp = sys.argv[3] if len(sys.argv) > 3 else None
    run_ab(labels, newp, bstp)
