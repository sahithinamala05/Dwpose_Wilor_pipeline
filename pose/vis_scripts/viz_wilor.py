"""Overlay WiLoR hands on the video. Reads a STAGE-2 pkl (baccarat/wilor/<stem>_wilor.pkl):
uses pose['hands'] (K,21,2 norm), pose['hands_is_right'], pose['hands_conf'].
right=green, left=red, joints=yellow; each hand labelled with its detector confidence.
Side-by-side orig | overlay -> <out>/<stem>_wilor.mp4. Run in the `wilor` conda env.

--min-conf keeps the overlay clean when the batch was run at a low DET_CONF (e.g. DET_CONF=0.0
produces 300 hands/frame -- raise --min-conf to only draw the confident ones).

Usage: python viz_wilor.py [--src vid] [--pkl wilor] [--out vis_wilor] [--procs 3]
                           [--min-conf 0.1] [--max-frames 0]
"""
import os, sys, glob, pickle, subprocess, argparse
import numpy as np, cv2; cv2.setNumThreads(1)
from multiprocessing import Pool

HAND_E = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
MIN_CONF = 0.1   # only draw hands with detector conf >= this (overridden by --min-conf)

def draw(img, pose, W, H):
    if pose is None:
        return 0
    hh = pose.get('hands')
    if hh is None or not len(np.asarray(hh)):
        return 0
    hh = np.asarray(hh)
    isr = list(pose['hands_is_right']); cf = list(pose['hands_conf'])   # required (no fallback)
    drawn = 0
    for k in range(hh.shape[0]):
        c = float(cf[k])
        if c < MIN_CONF: continue
        right = bool(isr[k])
        col = (0,255,0) if right else (0,0,255)                       # right=green, left=red
        pts = hh[k]*[W, H]
        for e in HAND_E:
            cv2.line(img, tuple(pts[e[0]].astype(int)), tuple(pts[e[1]].astype(int)), col, 2)
        for p in pts:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,220,255), -1)  # yellow joints
        w0 = pts[0]
        cv2.putText(img, f"{'R' if right else 'L'} {c:.2f}", (int(w0[0]), int(w0[1])-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        drawn += 1
    return drawn

def render(job):
    stem, vp, pk, out, maxf = job
    try:
        d = pickle.load(open(pk, 'rb'))
        cap = cv2.VideoCapture(vp); w = int(cap.get(3)); h = int(cap.get(4)); fps = cap.get(5)
        if w <= 0 or h <= 0 or fps <= 0:
            cap.release(); return f"ERR {stem}: unreadable video dims/fps ({w}x{h}@{fps})"
        ff = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24",
            "-s", f"{2*w}x{h}", "-r", str(int(fps)), "-i", "pipe:0", "-c:v","libx264","-pix_fmt","yuv420p",
            "-crf","23","-movflags","+faststart", f"{out}/{stem}_wilor.mp4"], stdin=subprocess.PIPE)
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok: break
            ann = img.copy()
            n = draw(ann, d[idx].get('pose') if idx < len(d) else None, w, h)
            cv2.putText(ann, f"f{idx}  WiLoR hands (R=green L=red, conf>={MIN_CONF})  n={n}",
                        (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            combo = np.hstack([img, ann])
            ff.stdin.write(np.ascontiguousarray(combo).tobytes()); idx += 1
            if maxf and idx >= maxf: break
        cap.release(); ff.stdin.close(); ff.wait(); return f"ok {stem}: {idx} frames"
    except Exception as e:
        return f"ERR {stem} {e}"

if __name__ == "__main__":
    B = "/home/ubuntu/us-west-3-fs/sahithi/baccarat"
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{B}/vid")
    ap.add_argument("--pkl", default=f"{B}/wilor")
    ap.add_argument("--out", default=f"{B}/vis/vis_wilor")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--min-conf", type=float, default=MIN_CONF, help="only draw hands with detector conf >= this")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames per clip (0=all; for quick previews)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of clips (0=all)")
    ap.add_argument("--stem", default=None, help="render only this clip id (e.g. 2025-12-15_19-24-28)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    MIN_CONF = a.min_conf
    jobs = []
    for pk in sorted(glob.glob(os.path.join(a.pkl, "*_wilor.pkl"))):
        stem = os.path.basename(pk)[:-len("_wilor.pkl")]
        if a.stem and stem != a.stem: continue
        vp = next((os.path.join(a.src, stem+e) for e in (".mkv", ".mp4")
                   if os.path.exists(os.path.join(a.src, stem+e))), None)
        if vp: jobs.append((stem, vp, pk, a.out, a.max_frames))
    if a.limit: jobs = jobs[:a.limit]
    print(f"rendering {len(jobs)} WiLoR clips -> {a.out}", flush=True)
    with Pool(a.procs) as p:
        for m in p.imap_unordered(render, jobs): print(" ", m, flush=True)
    print("DONE", flush=True)
