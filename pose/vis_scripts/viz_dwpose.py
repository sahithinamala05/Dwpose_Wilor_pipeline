"""Overlay DWPose output (body + face + DWPose hands) on the video. Reads a STAGE-1 pkl
(baccarat/dwpose/<stem>_dwpose.pkl). body=white limbs/joints, face=cyan dots,
hands: left=orange / right=green (joints=yellow). Hand points gated by hands_scores.
Side-by-side orig | overlay -> <out>/<stem>_dwpose.mp4. Run in the `wilor` conda env.

Usage: python viz_dwpose.py [--src vid] [--pkl dwpose] [--out vis_dwpose] [--procs 3]
                            [--min-hand 0.3] [--max-frames 0]
"""
import os, sys, glob, pickle, subprocess, argparse
import numpy as np, cv2; cv2.setNumThreads(1)
from multiprocessing import Pool

LIMBS = [[2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],[10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],[1,16],[16,18]]
HAND_E = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
MIN_HAND = 0.3   # per-point hand-score gate (overridden by --min-hand)

def draw(img, pose, W, H):
    if pose is None:
        return
    b = pose.get('bodies')
    if b is not None:
        cand = np.asarray(b['candidate']); sub = np.asarray(b['subset'])
        for pn in range(sub.shape[0]):
            for u, v in LIMBS:
                ia, ib = int(sub[pn, u-1]), int(sub[pn, v-1])
                if ia < 0 or ib < 0: continue
                pa = cand[ia]*[W, H]; pb = cand[ib]*[W, H]
                cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255,255,255), 2)
            for j in range(18):
                idx = int(sub[pn, j])
                if idx < 0: continue
                p = cand[idx]*[W, H]; cv2.circle(img, (int(p[0]), int(p[1])), 3, (255,255,255), -1)
    faces = pose.get('faces')
    if faces is not None:
        faces = np.asarray(faces)
        for pn in range(faces.shape[0]):
            for p in faces[pn]:
                if p[0] <= 0 and p[1] <= 0: continue
                cv2.circle(img, (int(p[0]*W), int(p[1]*H)), 1, (255,255,0), -1)       # cyan
    hh = pose.get('hands')
    if hh is not None and len(np.asarray(hh)):
        hh = np.asarray(hh); hs = np.asarray(pose['hands_scores'])                     # required (no fallback)
        nH = hh.shape[0]; half = nH // 2                                              # [0:N]=left, [N:2N]=right
        for k in range(nH):
            sc = hs[k]
            col = (0,255,0) if k >= half else (0,165,255)                              # right=green, left=orange
            pts = hh[k]*[W, H]
            for e in HAND_E:
                if sc[e[0]] < MIN_HAND or sc[e[1]] < MIN_HAND: continue
                cv2.line(img, tuple(pts[e[0]].astype(int)), tuple(pts[e[1]].astype(int)), col, 2)
            for ji, p in enumerate(pts):
                if sc[ji] < MIN_HAND: continue
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,220,255), -1)            # yellow joints

def render(job):
    stem, vp, pk, out, maxf = job
    try:
        d = pickle.load(open(pk, 'rb'))
        cap = cv2.VideoCapture(vp); w = int(cap.get(3)); h = int(cap.get(4)); fps = cap.get(5)
        if w <= 0 or h <= 0 or fps <= 0:
            cap.release(); return f"ERR {stem}: unreadable video dims/fps ({w}x{h}@{fps})"
        ff = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24",
            "-s", f"{2*w}x{h}", "-r", str(int(fps)), "-i", "pipe:0", "-c:v","libx264","-pix_fmt","yuv420p",
            "-crf","23","-movflags","+faststart", f"{out}/{stem}_dwpose.mp4"], stdin=subprocess.PIPE)
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok: break
            ann = img.copy()
            draw(ann, d[idx].get('pose') if idx < len(d) else None, w, h)
            cv2.putText(ann, f"f{idx}  DWpose: body(white) face(cyan) hands(L=orange R=green)",
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
    ap.add_argument("--pkl", default=f"{B}/dwpose")
    ap.add_argument("--out", default=f"{B}/vis/vis_dwpose")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--min-hand", type=float, default=MIN_HAND, help="per-point hand-score gate")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames per clip (0=all; for quick previews)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of clips (0=all)")
    ap.add_argument("--stem", default=None, help="render only this clip id (e.g. 2025-12-15_19-20-48)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    MIN_HAND = a.min_hand
    jobs = []
    for pk in sorted(glob.glob(os.path.join(a.pkl, "*_dwpose.pkl"))):
        stem = os.path.basename(pk)[:-len("_dwpose.pkl")]
        if a.stem and stem != a.stem: continue
        vp = next((os.path.join(a.src, stem+e) for e in (".mkv", ".mp4")
                   if os.path.exists(os.path.join(a.src, stem+e))), None)
        if vp: jobs.append((stem, vp, pk, a.out, a.max_frames))
    if a.limit: jobs = jobs[:a.limit]
    print(f"rendering {len(jobs)} DWpose clips -> {a.out}", flush=True)
    with Pool(a.procs) as p:
        for m in p.imap_unordered(render, jobs): print(" ", m, flush=True)
    print("DONE", flush=True)
