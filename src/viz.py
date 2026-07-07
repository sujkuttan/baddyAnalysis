import cv2
import numpy as np


SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def _draw_pose(frame, kp, color):
    kp = np.array(kp)
    for a, b in SKELETON:
        if a < len(kp) and b < len(kp) and not np.any(np.isnan(kp[a])) and not np.any(np.isnan(kp[b])):
            pa = tuple(int(v) for v in kp[a])
            pb = tuple(int(v) for v in kp[b])
            cv2.line(frame, pa, pb, color, 2)
            cv2.circle(frame, pa, 3, color, -1)


def draw_annotated_video(video_path, frames, Hs, shuttle_court, contact_frames, attrib,
                         preds, out_path="data/annotated.mp4", court_size=(240, 520)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw, vh = w + court_size[0] + 20, max(h, court_size[1] + 20)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
    cw, ch = court_size
    colors = {0: (0, 255, 0), 1: (0, 200, 255), 2: (255, 0, 0)}

    fi = 0
    contact_set = {c: (attrib[i], preds[i] if i < len(preds) else "?") for i, c in enumerate(contact_frames)}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi < len(frames):
            for d in frames[fi]:
                _draw_pose(frame, d["keypoints"], colors.get(d["id"] % 3, (255, 255, 255)))
        if fi < len(shuttle_court) and not np.any(np.isnan(shuttle_court[fi])):
            x, y = int(shuttle_court[fi][0]), int(shuttle_court[fi][1])
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        panel = np.zeros((ch, cw, 3), dtype=np.uint8)
        cv2.rectangle(panel, (5, 5), (cw - 5, ch - 5), (255, 255, 255), 1)
        out = np.zeros((vh, vw, 3), dtype=np.uint8)
        out[:h, :w] = frame
        out[:ch, w + 20:w + 20 + cw] = panel
        if fi in contact_set:
            pid, pred = contact_set[fi]
            cv2.putText(out, f"{pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        writer.write(out)
        fi += 1
    cap.release()
    writer.release()
    return out_path


def plot_heatmap(heatmap, out_path="data/coverage_heatmap.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 8))
    plt.imshow(heatmap.T, origin="lower", cmap="hot")
    plt.axis("off")
    plt.title("Court Coverage")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_fatigue(fatigue, out_path="data/fatigue.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pids = list(fatigue.keys())
    vals = [fatigue[p]["fatigue_score"] for p in pids]
    plt.figure(figsize=(5, 3))
    plt.bar([str(p) for p in pids], vals)
    plt.title("Fatigue score (higher = more decline)")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path
