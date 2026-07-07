# Badminton Stroke & Performance Analysis Pipeline

Standalone pipeline for **hand-held phone match videos** of advanced sub-junior players.
It fixes the core failure of broadcast-trained models (BST/LSTM/ST-GCN) on shaky,
variable-viewpoint footage by changing the *feature representation*, not the model.

## Core idea — Stabilized Multi-Trajectory Fusion
Phone footage corrupts video features via camera shake and tiny players. The pipeline:

- **Pillar A (stabilizer):** tracks court-line points every frame (KLT + RANSAC) and warps
  all detections into a stable court-coordinate frame. Fixes attribution and de-noises trajectories.
- **Pillar B (racket):** bootstraps a racket-tip trajectory from wrist extrapolation (no labeling).
- **Pillar C (dense MBH):** camera-invariant motion descriptors (RAFT/Farneback + MBH),
  region-constrained to the player.
- A **fusion classifier** late-fuses normalized pose + racket trajectory + MBH, trained on
  *your* labeled footage. A geometry baseline runs even before training.

Outputs: stroke classification + player attribution, footwork / court coverage / fatigue
analytics, an LLM coaching report, an annotated video, and an A/B comparison vs your BST pipeline.

## Install (Colab / local)
```
pip install -r requirements.txt
```

## Court corners
Create `corners.json` (click 4 court corners in order TL, TR, BR, BL, e.g. with `src.stabilize.interactive_select_corners`):
```json
{"corners": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}
```

## Run the full pipeline
```
python run.py pipeline --video match.mp4 --corners corners.json \
    --labels labels_import.csv --out data --tracknet weights/TrackNet_best.pt \
    --device cuda
```
- Produces `data/new_predictions.csv`, `data/metrics.json`, `data/annotated.mp4`,
  `data/coverage_heatmap.png`, `data/fatigue.png`, `data/coaching_report.md`.
- If `labels_import.csv` is provided it fine-tunes the fusion classifier on your labeled shots;
  otherwise a geometry baseline is used.

## A/B vs your BST pipeline
BST predictions on the 100 labeled shots must be supplied (the `labeled` rows in
`labels_import.csv` have blank `predicted_stroke`). Export your BST predictions keyed by `frame`:
```
python run.py ab --labels labels_import.csv --new data/new_predictions.csv --bst bst_predictions.csv
```
This prints accuracy + confusion matrices for BST and the new pipeline on the shared 100 labeled shots.

## Label format
`labels_import.csv` columns: `shot_id, frame, ts_start, ts_end, player_id, side,
predicted_stroke, predicted_class_id, true_stroke, true_class_id, label_status, source`.
Labeled rows have `label_status=='labeled'` and `true_stroke`/`true_class_id` filled.
Canonical stroke vocabulary (11 classes) is defined in `src/config.py`.

## Module map
| File | Responsibility |
|---|---|
| `src/stabilize.py` | per-frame homography (Pillar A) |
| `src/pose.py` | YOLOv8 + ByteTrack tracking & pose, court warp, foot point |
| `src/shuttle.py` | TrackNetV3 shuttle tracking |
| `src/contact.py` | contact-frame detection + rally segmentation |
| `src/racket_bootstrap.py` | racket-tip trajectory (Pillar B) |
| `src/dense_traj.py` | RAFT/Farneback + MBH (Pillar C) |
| `src/biomech.py` | kinetic chain + contact→player attribution |
| `src/classifier.py` | fusion classifier + stroke-window features |
| `src/baseline.py` | geometry stroke baseline (pre-training) |
| `src/movement.py` | coverage, speed, distance, heatmap, fatigue |
| `src/llm_feedback.py` | LLM coaching report |
| `src/viz.py` | annotated video + charts |
| `src/ab_eval.py` | A/B comparison harness |

## Known limitations
- A/B join is on `frame`; one labeled frame (2408) is duplicated → ~1% harness artifact.
- Racket bootstrap is a proxy; fine-tuning a detector (RacketVision) improves it.
- Fusion classifier needs more cross-video labels to exceed BST reliably — use the labeling loop.
