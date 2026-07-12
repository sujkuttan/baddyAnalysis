import json

cells = []
def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})

def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})

md('''# Badminton Analysis Pipeline (Colab)
Runs the stabilized multi-trajectory pipeline on your hand-held phone match videos.

**Before running:** `Runtime ▸ Change runtime type ▸ GPU` (T4) for speed. The pipeline
auto-falls back to CPU if CUDA is unavailable and processes the video in frame batches
to stay within Colab's 16 GB GPU / 12 GB CPU limits.''')

md('''## 1. Clone + install''')
code('''!git clone https://github.com/sujkuttan/baddyAnalysis.git
%cd baddyAnalysis
!pip install -r requirements.txt''')

md('''## 2. Download match video from Google Drive
Edit `VIDEO_ID` if the link changes. The file id is the part after `file/d/` in the share URL.''')
code('''!pip install -q gdown
VIDEO_ID = "1aA_3keNIfCBjkNC9isovHwTQjfVejQdC"
!gdown {VIDEO_ID} -O match.mp4
video_name = "match.mp4"
print("video:", video_name)''')

md('''## 3. Court corners
The first frame is shown for reference. **Paste your 4 court corners in order:
TL, TR, BR, BL** (x,y in pixels). They are saved to `corners.json`.''')
code('''import cv2, os, json
from matplotlib import pyplot as plt

os.makedirs('data', exist_ok=True)
cap = cv2.VideoCapture(video_name); ret, frame = cap.read(); cap.release()
if not ret:
    raise RuntimeError('could not read first frame from ' + video_name)
cv2.imwrite('data/first_frame.jpg', frame)
print('saved data/first_frame.jpg; image size (w,h):', frame.shape[1], frame.shape[0])

img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 6))
plt.imshow(img); plt.axis('on'); plt.title('Verify court corners (TL, TR, BR, BL)')
plt.show()

# PASTE corners in order TL, TR, BR, BL
corners = [
    [466, 77],
    [831, 80],
    [1181, 641],
    [148, 637],
]
assert len(corners) == 4, 'need exactly 4 corners (TL, TR, BR, BL)'
json.dump({'corners': corners}, open('corners.json', 'w'))
print('saved corners.json:', corners)''')

md('''## 4. Download TrackNet weights (zip) from Google Drive
The zip includes TrackNet + inpaint weights. It is unzipped into `weights/` and the
`.pt` is located automatically.''')
code('''import os, glob
!gdown 1rhKXbff1GITgrFTYptW6gAvWZ76E_qzp -O tracknet.zip
!unzip -o tracknet.zip -d weights/
cands = glob.glob('weights/**/TrackNet*.pt', recursive=True)
tracknet = cands[0] if cands else None
icands = glob.glob('weights/**/Inpaint*.pt', recursive=True)
inpaintnet = icands[0] if icands else None
print('tracknet weights:', tracknet)
print('inpaintnet weights:', inpaintnet)
print('weights dir:', os.listdir('weights'))''')

md('''## 5. Run the pipeline
Processes the video in batches of `BATCH_SIZE` frames (default 64 on T4) so the full
match runs within Colab's RAM. TrackNet's internal `chunk=8` is the main GPU-memory
knob; if Colab CUDA OOMs, lower TrackNet's internal chunk from 8 to 4 in
`src/shuttle.py`. `SAMPLE_FRAMES` limits processing for a quick test; set to `None`
for the entire 5-min sample.

**Frame budget (video is 30 fps):**
- 900 frames ≈ 30 s (fast smoke test, no labels in range → baseline only)
- ~3600 frames ≈ 120 s (covers all labeled shots → trains the fusion classifier)
- `None` = full 5-min sample (≈9000 frames)''')
code('''import torch, json
from src import pipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device:', device)

corners = json.load(open('corners.json'))['corners']

BATCH_SIZE = 64
# 900 = ~30s smoke test. 3600 = ~120s (includes all labeled shots, trains classifier).
# None = full 5-min sample. The 100 labels are on the first ~120s of this sample.
SAMPLE_FRAMES = 3600
# Pose model: default yolov8s-pose.pt. Use a larger model (e.g. yolov8m-pose.pt) or a
# newer architecture for better detection of small/distant players. POSE_UPSCALE
# enlarges each frame before pose detection (more pixels on distant players).
POSE_MODEL = 'yolov8l-pose.pt'
POSE_UPSCALE = 2.0
POSE_CONF = 0.25
# Known player count: 2 for singles, 4 for doubles. Forces fragmented tracker IDs
# to merge into exactly this many players so one athlete is not split into several
# phantom tracks (which breaks contact attribution). The sample video is singles.
MAX_PLAYERS = 2
# --- TrackNet resolution levers (A/B/C sweep) ---
# A (img_size): TrackNet input resolution (H,W). Official weights train at 288x512.
# Raising this (e.g. (384,680), both MUST be divisible by 8 -- the U-Net pools /8)
# gives the far shuttle more pixels globally at higher GPU cost. Keep (288,512)
# to stay in-distribution. (384,680) ~2.1x pixels; (384,688) also valid.
TRACKNET_IMG_SIZE = (288, 512)
# B (court crop): crop each frame to the court region before the resize (more
# pixels on the distant far shuttle). OFF -- the sample video already fills the
# frame, so this is a no-op here; useful only for wide-angle footage.
TRACKNET_CROP = False
# C (far-court tile): second TrackNet pass on the far court (+headroom) for ~2x
# pixels on the far shuttle; used only to FILL GAPS the full-frame pass missed
# (never overrides a confident detection). Flip True for an A/B run.
FAR_TILE = False
print(f'--- RUN: batch_size={BATCH_SIZE}, sample_frames={SAMPLE_FRAMES}, '
      f'pose_model={POSE_MODEL}, pose_upscale={POSE_UPSCALE}, pose_conf={POSE_CONF}, '
      f'max_players={MAX_PLAYERS}, tracknet_img_size={TRACKNET_IMG_SIZE}, '
      f'tracknet_crop={TRACKNET_CROP}, far_tile={FAR_TILE} ---')
res = pipeline.run_full_pipeline(
    video_name, corners, out_dir='data',
    labels_csv='labels_import.csv', device=device,
    tracknet_weights=tracknet, inpaintnet_weights=inpaintnet, batch_size=BATCH_SIZE,
    max_frames=SAMPLE_FRAMES, debug=True, max_players=MAX_PLAYERS,
    pose_model=POSE_MODEL, pose_upscale=POSE_UPSCALE, pose_conf=POSE_CONF,
    tracknet_crop=TRACKNET_CROP, far_tile=FAR_TILE, tracknet_img_size=TRACKNET_IMG_SIZE,
)
print('predictions:', res['predictions_csv'])
print('metrics:', res['metrics'])''')

md('''## 6. A/B vs your BST pipeline
Export your BST predictions to `bst_predictions.csv` (columns `frame,predicted_stroke`),
upload it via `files.upload()`, then compare. The `labeled` rows in `labels_import.csv`
are the shared ground truth. If you don't have BST preds yet, skip this cell.''')
code('''from google.colab import files
# files.upload()  # upload bst_predictions.csv  (optional)
from src import pipeline
bst = 'bst_predictions.csv' if os.path.exists('bst_predictions.csv') else None
pipeline.ab_compare('labels_import.csv', 'data/new_predictions.csv', bst)''')

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "colab": {"name": "baddyAnalysis_colab.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
json.dump(nb, open('notebooks/baddyAnalysis_colab.ipynb', 'w'), indent=1)
print('wrote notebook with', len(cells), 'cells')
