# AGENTS.md — baddyAnalysis

Badminton stroke & performance analysis pipeline for hand-held phone match videos.
Standalone repo at `github.com/sujkuttan/baddyAnalysis` (branch `main`). No tests, no CI, no lint config. Verify changes with `python3 -m py_compile src/*.py run.py gen_nb.py`.

## Commands
- Generate the Colab notebook (required — `notebooks/*.ipynb` are gitignored, not source): `python3 gen_nb.py` → writes `notebooks/baddyAnalysis_colab.ipynb`.
- Local/CLI full pipeline: `python run.py pipeline --video match.mp4 --corners corners.json --labels labels_import.csv --out data --tracknet weights/TrackNet_best.pt --device cuda`.
- A/B vs BST: `python run.py ab --labels labels_import.csv --new data/new_predictions.csv --bst bst_predictions.csv`.
- Real runs happen in **Google Colab (T4 GPU)** via the generated notebook, not locally. `gen_nb.py` is the source of truth for the notebook.

## Architecture & entrypoints
- `src/pipeline.py` `run_full_pipeline` is the orchestrator (batching loop → stabilize → pose → TrackNet → InpaintNet → contacts → classifier → metrics → viz → report).
- Frame processing is **batched (default `batch_size=128`)** and raw frames are discarded per batch to fit Colab's 16GB GPU / 12GB RAM. TrackNet runs on GPU; running the full video with pose on GPU caused CUDA OOM before batching.
- `src/config.py`: court constants (`COURT_WIDTH=6.1`, `COURT_LENGTH=13.4`), canonical 11-class stroke vocab, and `validate_court_corners` (raises if corners not in TL,TR,BR,BL order).
- Module map: `stabilize` (per-frame homography Pillar A), `pose` (YOLOv8-pose+ByteTrack → court warp), `shuttle` (TrackNetV3), `inpaintnet` (TrackNetV3 trajectory rectifier), `contact`, `racket_bootstrap`, `dense_traj`, `biomech`, `classifier`, `baseline`, `movement`, `llm_feedback`, `viz`, `ab_eval`.

## Critical model quirks (easy to break)
- **TrackNetV3** (`src/shuttle.py`): input is **27 channels = 9 stacked RGB frames**; output is **8 heatmaps**; `_preprocess` resizes frames to `img_size=(288,512)` then reshapes to 27ch, and coords are scaled back to the original frame size. Weights must load `104/104` `strict=True` — a mismatch means the model is untrained (prints a WARNING). Model is a U-Net (skip connections) with `nn.Upsample`+`Conv2d` up blocks (NOT ConvTranspose2d).
- **InpaintNet** (`src/inpaintnet.py`) is NOT image inpainting — it is TrackNetV3's **trajectory rectification network** (1D-conv U-Net). Input `(N,L,2)` normalized coords + `(N,L,1)` mask; output repairs only `mask==1` frames (`out = pred*mask + coor*(1-mask)`). The checkpoint is a training wrapper: weights live under `sd['model']`; the bottleneck layer is spelled `buttelneck` in code but the checkpoint may have `buttleneck` — the loader normalizes both and strips `module.` prefixes. MUST load `18/18` keys.
- `inpaintnet.load_inpaintnet(path)` returns `None` on any failure (so `pipeline.py` skips rectification). A `None` weight path is normal — it is not an error.

## Data & labels
- Court corners are `[[466,77],[831,80],[1181,641],[148,637]]` (TL,TR,BR,BL) for the sample video; `validate_court_corners` will raise if reordered wrong.
- `labels_import.csv`: labeled rows have `label_status=='labeled'` (~99 shots on the first ~120s / ~3600 frames). A/B join is on `frame`; one frame (2408) is duplicated in labels (~1% harness artifact, harmless).
- Video is **30 fps**. Frame budget: 900≈30s (no labels), 3600≈120s (all labels, trains classifier), `None`=full 5-min sample (≈9000 frames). The labeled range is only the first ~120s.

## Known open issues (do not "fix" blindly)
- **Shuttle coverage** (`shuttle_nonnan`) has been unstable across Colab runs (68% → 41% → 31%). `rectify_trajectory` provably preserves TrackNet-valid points in local repro, so the Colab drop is unresolved. `pipeline.py` prints `[diag] tracknet_valid / rectified_valid / pre_oob / oob_clipped` — use it to localize loss (rectify vs warp/OOB) before changing code.
- `detect_contacts_near_players` uses `max_dist=2.0`; fusion classifier match window is `MATCH_WINDOW=15` frames and `labeled_matched@3` is low (~18/99) — stroke counts need work.
- A `266 m/s` max-shuttle-speed outlier (in-bounds teleport) is NOT yet filtered — only *misses* are rectified, not wild *detections*.
- Movement smoothing (`_smooth`, median win=3) + `max_step=0.8 m/frame` (=24 m/s cap) were tightened to kill speed outliers; don't loosen without re-checking speed metrics.

## Conventions
- `.gitignore` excludes `weights/`, `data/`, `*.pt`, `*.ipynb_checkpoints/`, and `notebooks/*.ipynb` outputs. Keep large artifacts out of the repo.
- Drive assets: video id `1aA_3keNIfCBjkNC9isovHwTQjfVejQdC`, TrackNet zip `1rhKXbff1GITgrFTYptW6gAvWZ76E_qzp` (contains `weights/ckpts/TrackNet_best.pt` + `InpaintNet_best.pt`).
