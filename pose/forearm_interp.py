"""Confident forearm (wrist + elbow) with View-2 interpolation.

Per frame, 4 channels: L-elbow, R-elbow (DWpose body, best person) and L-wrist, R-wrist
(WiLoR hand root). A point is kept only if its confidence >= --conf (default 0.70); elbows must
also be in-frame (an off-frame elbow coordinate isn't usable). For VIEW-2 segments only, any
frame whose point is missing is LINEARLY INTERPOLATED between the nearest confident frames on
each side (interior gaps only; segment-edge gaps stay missing). View-1 frames are left as the
raw confident points (no interpolation).

Reads dwpose + wilor pkls + the temporal views JSON. Output per clip:
  <out>/<clip>_forearm.json   frames[i] = {view, pts:{name:[x,y,src]}}  src in {conf, interp, miss}
Optional --viz overlays the forearm (elbow->wrist per side) on the video (green=confident,
yellow=interpolated) for View-2. Run in the `wilor` conda env.

Usage:
  python forearm_interp.py --stem 2025-12-15_19-20-48 --viz
  python forearm_interp.py --limit 100 --procs 4
"""
import os, sys, glob, json, pickle, argparse, subprocess
import numpy as np, cv2; cv2.setNumThreads(1)
from multiprocessing import Pool

B = "/home/ubuntu/us-west-3-fs/sahithi/baccarat"
DW, WL, VIEWS, VID = f"{B}/dwpose", f"{B}/wilor", f"{B}/temporal_annotation", f"{B}/vid"
CH = ["L-elbow", "R-elbow", "L-wrist", "R-wrist"]
ELB = {"L-elbow": 6, "R-elbow": 3}              # openpose body idx: L-elbow=6, R-elbow=3
PAIRS = [("L-elbow", "L-wrist"), ("R-elbow", "R-wrist")]
CONF = 0.70
ELBOW_INFRAME = True       # if False, keep confident elbows even when off-frame (arm line runs to the edge)


def elbow_pts(dwframe, W, H, conf):
    """best-person elbows with score >= conf AND in-frame -> {name:(x,y)} (missing keys omitted)."""
    out = {}
    p = dwframe.get('pose') if isinstance(dwframe, dict) else None
    if not isinstance(p, dict) or p.get('bodies') is None:
        return out
    b = p['bodies']; cand = np.asarray(b['candidate']); sub = np.asarray(b['subset']); sc = np.asarray(b['scores'])
    if sub.shape[0] == 0:
        return out
    means = []
    for pn in range(sub.shape[0]):
        vals = [sc[pn*18 + j] for j in range(18) if sub[pn, j] >= 0]
        means.append(np.mean(vals) if vals else 0.0)
    pn = int(np.argmax(means))
    for name, j in ELB.items():
        idx = int(sub[pn, j])
        if idx < 0:
            continue
        s = sc[pn*18 + j]; x, y = cand[idx] * [W, H]
        if s >= conf and (not ELBOW_INFRAME or (0 <= x < W and 0 <= y < H)):
            out[name] = (float(x), float(y))
    return out


def wrist_pts(wlframe, W, H, conf):
    """highest-conf WiLoR hand per side with conf >= conf -> {name:(x,y)} (kpt0 = wrist)."""
    out = {}; best = {"L-wrist": -1.0, "R-wrist": -1.0}
    p = wlframe.get('pose') if isinstance(wlframe, dict) else None
    if not isinstance(p, dict) or p.get('hands') is None:
        return out
    hands = np.asarray(p['hands']); isr = list(p.get('hands_is_right', [])); cf = list(p.get('hands_conf', []))
    for k in range(len(hands)):
        name = "R-wrist" if (k < len(isr) and isr[k]) else "L-wrist"
        c = float(cf[k]) if k < len(cf) else 0.0
        if c >= conf and c > best[name]:
            best[name] = c; wr = hands[k][0] * [W, H]; out[name] = (float(wr[0]), float(wr[1]))
    return out


def compute(cid, conf):
    dw = pickle.load(open(f"{DW}/{cid}_dwpose.pkl", "rb"))
    wl = pickle.load(open(f"{WL}/{cid}_wilor.pkl", "rb"))
    vj = json.load(open(f"{VIEWS}/{cid}_views.json"))
    N = min(len(dw), len(wl)); H, W = dw[0]['frame_dimensions']
    view = np.ones(N, int)
    for s in vj['segments']:
        view[max(0, s['start']):min(N, s['end']+1)] = s['view']

    pts = {ch: [None]*N for ch in CH}          # (x,y) or None
    src = {ch: ['miss']*N for ch in CH}
    for i in range(N):
        got = {**elbow_pts(dw[i], W, H, conf), **wrist_pts(wl[i], W, H, conf)}
        for ch in CH:
            if ch in got:
                pts[ch][i] = got[ch]; src[ch][i] = 'conf'

    # interpolate interior gaps within each VIEW-2 segment
    n_interp = 0
    for s in vj['segments']:
        if s['view'] != 2:
            continue
        a, b = max(0, s['start']), min(N-1, s['end'])
        for ch in CH:
            valid = [i for i in range(a, b+1) if pts[ch][i] is not None]
            for v0, v1 in zip(valid, valid[1:]):
                if v1 - v0 <= 1:
                    continue
                p0, p1 = pts[ch][v0], pts[ch][v1]
                for t in range(v0+1, v1):
                    f = (t - v0) / (v1 - v0)
                    pts[ch][t] = (p0[0] + (p1[0]-p0[0])*f, p0[1] + (p1[1]-p0[1])*f)
                    src[ch][t] = 'interp'; n_interp += 1

    frames = []
    for i in range(N):
        fr = {"view": int(view[i]), "pts": {}}
        for ch in CH:
            p = pts[ch][i]
            fr["pts"][ch] = [round(p[0], 2), round(p[1], 2), src[ch][i]] if p is not None else [0, 0, 'miss']
        frames.append(fr)
    rep = dict(clip=cid, n_frames=N, conf=conf, channels=CH, frames=frames,
               n_interp=n_interp,
               conf_counts={ch: sum(1 for i in range(N) if src[ch][i] == 'conf') for ch in CH})
    return rep, (W, H)


def draw_before(img, pose, W, H):
    """BEFORE = raw DWpose forearm: elbow->wrist for EVERY detected person, unthresholded."""
    if not isinstance(pose, dict) or pose.get('bodies') is None:
        return
    b = pose['bodies']; cand = np.asarray(b['candidate']); sub = np.asarray(b['subset'])
    for pn in range(sub.shape[0]):
        for elb, wri in [(3, 4), (6, 7)]:              # R / L forearm (openpose idx)
            ia, ib = int(sub[pn, elb]), int(sub[pn, wri])
            if ia < 0 or ib < 0:
                continue
            pa = cand[ia]*[W, H]; pb = cand[ib]*[W, H]
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 255, 255), 2)
            cv2.circle(img, (int(pa[0]), int(pa[1])), 5, (255, 0, 255), -1)   # elbow magenta
            cv2.circle(img, (int(pb[0]), int(pb[1])), 5, (255, 0, 255), -1)   # wrist magenta


def render(cid, rep, out_mp4):
    cap = cv2.VideoCapture(f"{VID}/{cid}.mp4"); w = int(cap.get(3)); h = int(cap.get(4)); rr = cap.get(5)
    if w <= 0 or h <= 0 or rr <= 0:
        cap.release(); raise RuntimeError("bad video dims")
    dw = pickle.load(open(f"{DW}/{cid}_dwpose.pkl", "rb"))     # raw DWpose for the BEFORE panel
    ff = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24",
        "-s", f"{2*w}x{h}", "-r", str(int(rr)), "-i", "pipe:0", "-c:v","libx264","-pix_fmt","yuv420p",
        "-crf","23","-movflags","+faststart", out_mp4], stdin=subprocess.PIPE)
    COL = {'conf': (0,255,0), 'interp': (0,255,255)}     # green / yellow
    i = 0; written = 0; F = rep["frames"]
    while True:
        ok, img = cap.read()
        if not ok: break
        if i < len(F) and F[i]["view"] == 2:             # VIEW-2 ONLY: skip view-1 frames
            left = img.copy(); ann = img.copy(); P = F[i]["pts"]
            draw_before(left, dw[i].get('pose') if i < len(dw) else None, w, h)   # BEFORE: raw DWpose
            for elb, wri in PAIRS:                                                # AFTER: conf + interp
                pe, pw = P[elb], P[wri]
                if pe[2] != 'miss' and pw[2] != 'miss':
                    c = COL.get('interp') if 'interp' in (pe[2], pw[2]) else COL['conf']
                    cv2.line(ann, (int(pe[0]), int(pe[1])), (int(pw[0]), int(pw[1])), c, 3)
                for pt in (pe, pw):
                    if pt[2] != 'miss':
                        cv2.circle(ann, (int(pt[0]), int(pt[1])), 6, COL.get(pt[2], (0,255,0)), -1)
            F2 = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(left, f"f{i}  BEFORE  raw DWpose forearm", (10, 30), F2, 0.8, (255, 255, 255), 2)
            cv2.putText(ann, f"AFTER  conf>={rep['conf']}+interp (green/yellow)", (10, 30),
                        F2, 0.8, (255, 255, 255), 2)
            ff.stdin.write(np.ascontiguousarray(np.hstack([left, ann])).tobytes()); written += 1
        i += 1
    cap.release(); ff.stdin.close(); ff.wait()
    return written


def run_one(job):
    cid, conf, out, viz, vizout = job
    try:
        rep, _ = compute(cid, conf)
        json.dump(rep, open(os.path.join(out, f"{cid}_forearm.json"), "w"))
        line = (f"  {cid}: interp={rep['n_interp']} conf_counts=" +
                ",".join(f"{k.split('-')[0][0]}{k.split('-')[1][0]}:{v}" for k, v in rep['conf_counts'].items()))
        if viz:
            if not any(f["view"] == 2 for f in rep["frames"]):
                line += " (no view-2, skip viz)"
            else:
                os.makedirs(vizout, exist_ok=True)
                w = render(cid, rep, os.path.join(vizout, f"{cid}_forearm.mp4"))
                line += f" + viz({w}f view2)"
        return line
    except Exception as e:
        return f"  ERR {cid}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{B}/forearm")
    ap.add_argument("--viz-out", default=f"{B}/vis/vis_forearm")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--allow-offframe-elbow", action="store_true",
                    help="keep confident elbows even when off-frame, so the forearm line is drawn to the edge")
    ap.add_argument("--stem", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ELBOW_INFRAME = not a.allow_offframe_elbow
    cids = sorted(os.path.basename(p)[:-len("_views.json")] for p in glob.glob(f"{VIEWS}/*_views.json"))
    cids = [c for c in cids if os.path.exists(f"{DW}/{c}_dwpose.pkl") and os.path.exists(f"{WL}/{c}_wilor.pkl")]
    if a.stem: cids = [c for c in cids if c == a.stem]
    if a.limit: cids = cids[:a.limit]
    jobs = [(c, a.conf, a.out, a.viz, a.viz_out) for c in cids]
    print(f"forearm interp: {len(jobs)} clip(s) conf>={a.conf} procs={a.procs} -> {a.out}", flush=True)
    if a.procs > 1 and len(jobs) > 1:
        with Pool(a.procs) as p:
            for m in p.imap_unordered(run_one, jobs): print(m, flush=True)
    else:
        for j in jobs: print(run_one(j), flush=True)
    print("DONE", flush=True)
