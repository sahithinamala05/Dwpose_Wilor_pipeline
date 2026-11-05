"""View-aware DWpose overlay, driven by the temporal face-visibility segmentation.

  VIEW 1 (face visible)            -> FULL DWpose: body (white), face (cyan), hands (L=orange/R=green)
  VIEW 2 (face hidden / card shot) -> ONLY hands + the elbow<->wrist (forearm) points (magenta joints)

Per-frame view comes from the segmentation JSON in baccarat/temporal_annotation/<stem>_views.json.
The JSON is REQUIRED -- a clip without one is SKIPPED (no inline fallback).
Reads the STAGE-1 DWpose pkls. Side-by-side orig | overlay -> baccarat/vis_dwpose/<stem>_dwpose.mp4.
Run in the `wilor` conda env.

Usage:
  python viz_dwpose_byview.py --limit 100                    # first 100 dwpose pkls
  python viz_dwpose_byview.py --stem 2025-12-15_19-20-48     # one clip
Options: --pkl-dir --views-dir --src --out --procs --max-frames --min-hand
"""
import os, sys, glob, json, pickle, subprocess, argparse
import numpy as np, cv2; cv2.setNumThreads(1)
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

LIMBS = [[2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],[10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],[1,16],[16,18]]
HAND_E = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
FOREARM = [(3, 4), (6, 7)]      # (elbow, wrist), 0-indexed openpose: right=(3,4) left=(6,7)
MIN_HAND = 0.3


def draw_hands(img, pose, W, H):
    hh = pose.get('hands')
    if hh is None or not len(np.asarray(hh)):
        return
    hh = np.asarray(hh); hs = np.asarray(pose['hands_scores'])    # required (no fallback)
    nH = hh.shape[0]; half = nH // 2                              # [0:N]=left, [N:2N]=right
    for k in range(nH):
        sc = hs[k]
        col = (0, 255, 0) if k >= half else (0, 165, 255)
        pts = hh[k] * [W, H]
        for e in HAND_E:
            if sc[e[0]] < MIN_HAND or sc[e[1]] < MIN_HAND: continue
            cv2.line(img, tuple(pts[e[0]].astype(int)), tuple(pts[e[1]].astype(int)), col, 2)
        for ji, p in enumerate(pts):
            if sc[ji] < MIN_HAND: continue
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 220, 255), -1)


def draw_full(img, pose, W, H):
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
                cv2.circle(img, (int(p[0]*W), int(p[1]*H)), 1, (255,255,0), -1)
    draw_hands(img, pose, W, H)


def draw_forearm_hands(img, pose, W, H):
    """VIEW 2: only the elbow<->wrist segment per arm + the hands."""
    b = pose.get('bodies')
    if b is not None:
        cand = np.asarray(b['candidate']); sub = np.asarray(b['subset'])
        for pn in range(sub.shape[0]):
            for elb, wri in FOREARM:
                ia, ib = int(sub[pn, elb]), int(sub[pn, wri])
                if ia < 0 or ib < 0: continue
                pa = cand[ia]*[W, H]; pb = cand[ib]*[W, H]
                cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255,255,255), 2)
                cv2.circle(img, (int(pa[0]), int(pa[1])), 5, (255,0,255), -1)   # elbow (magenta)
                cv2.circle(img, (int(pb[0]), int(pb[1])), 5, (255,0,255), -1)   # wrist (magenta)
    draw_hands(img, pose, W, H)


def view_per_frame(views_json, n):
    """Per-frame view (1/2) from the segmentation JSON. Returns None if the JSON is missing."""
    if not (views_json and os.path.exists(views_json)):
        return None                                   # REQUIRED: no inline fallback
    v = np.ones(n, dtype=int)
    d = json.load(open(views_json))
    for s in d['segments']:
        a = max(0, s['start']); b = min(n - 1, s['end'])
        v[a:b+1] = s['view']
    return v


def render(job):
    stem, vp, pk, views_json, out_dir, fps, maxf = job
    try:
        d = pickle.load(open(pk, 'rb'))
        v = view_per_frame(views_json, len(d))
        if v is None:
            return f"  SKIP {stem}: no views JSON -> {views_json}"
        cap = cv2.VideoCapture(vp); w = int(cap.get(3)); h = int(cap.get(4)); rr = cap.get(5)
        if w <= 0 or h <= 0 or rr <= 0:
            cap.release(); return f"  ERR {stem}: unreadable video dims/fps ({w}x{h}@{rr})"
        ff = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24",
            "-s", f"{2*w}x{h}", "-r", str(int(rr)), "-i", "pipe:0", "-c:v","libx264","-pix_fmt","yuv420p",
            "-crf","23","-movflags","+faststart", os.path.join(out_dir, f"{stem}_dwpose.mp4")], stdin=subprocess.PIPE)
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok: break
            ann = img.copy()
            pose = d[idx].get('pose') if idx < len(d) else None
            view = int(v[idx]) if idx < len(v) else 1
            if pose is not None:
                (draw_full if view == 1 else draw_forearm_hands)(ann, pose, w, h)
            lbl = "VIEW 1  full pose" if view == 1 else "VIEW 2  hands + forearm only"
            col = (0, 200, 0) if view == 1 else (0, 0, 255)
            cv2.putText(ann, f"f{idx}  {lbl}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            combo = np.hstack([img, ann])
            ff.stdin.write(np.ascontiguousarray(combo).tobytes()); idx += 1
            if maxf and idx >= maxf: break
        cap.release(); ff.stdin.close(); ff.wait(); return f"  ok {stem}: {idx}f"
    except Exception as e:
        return f"  ERR {stem} {e}"


if __name__ == "__main__":
    B = "/home/ubuntu/us-west-3-fs/sahithi/baccarat"
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-dir", default=f"{B}/dwpose")
    ap.add_argument("--views-dir", default=f"{B}/temporal_annotation")
    ap.add_argument("--src", default=f"{B}/vid")
    ap.add_argument("--out", default=f"{B}/vis/vis_dwpose")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stem", default=None, help="render only this clip id")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames per clip (preview)")
    ap.add_argument("--min-hand", type=float, default=MIN_HAND, help="per-point hand-score gate")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); MIN_HAND = a.min_hand
    jobs = []
    for pk in sorted(glob.glob(os.path.join(a.pkl_dir, "*_dwpose.pkl"))):
        stem = os.path.basename(pk)[:-len("_dwpose.pkl")]
        if a.stem and stem != a.stem: continue
        vp = next((os.path.join(a.src, stem+e) for e in (".mp4", ".mkv")
                   if os.path.exists(os.path.join(a.src, stem+e))), None)
        if vp:
            jobs.append((stem, vp, pk, os.path.join(a.views_dir, f"{stem}_views.json"),
                         a.out, 30.0, a.max_frames))
    if a.limit: jobs = jobs[:a.limit]
    print(f"view-aware DWpose viz: {len(jobs)} clip(s) procs={a.procs} -> {a.out}", flush=True)
    if a.procs > 1 and len(jobs) > 1:
        with Pool(a.procs) as p:
            for m in p.imap_unordered(render, jobs): print(m, flush=True)
    else:
        for j in jobs: print(render(j), flush=True)
    print("DONE", flush=True)
