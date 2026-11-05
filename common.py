"""Shared helpers for the baccarat DWpose/WiLoR analysis scripts.

Centralizes what was copy-pasted across forearm_interp / redo_interp / build_round_artifacts /
build_view_labels / segment_rounds: dataset paths, skeleton topology, DWpose-body / WiLoR-hand
extraction, view-run interpolation, view-segmentation loading, ffmpeg writing, and the table
card-presence signal. Import as:  import common as C   (or: from common import ...).

The inference core (stage1_dwpose, wilor_core, stage2_wilor_hands, batch_pipeline) does NOT depend
on this module.
"""
import os, json, pickle, subprocess
import numpy as np, cv2
cv2.setNumThreads(1)

# ----------------------------- paths -----------------------------
# dataset root — override with the BACCARAT_ROOT env var to run elsewhere
B          = os.environ.get("BACCARAT_ROOT", "/home/ubuntu/us-west-3-fs/sahithi/baccarat")
VID        = f"{B}/vid"
DWPOSE     = f"{B}/dwpose"
WILOR      = f"{B}/wilor"
VIEW_ANNOT = f"{B}/view_annot"
VIS        = f"{B}/vis"
ANNOT      = f"{B}/annotation"
DEALERS    = f"{B}/dealers"
FPS        = 30

# ----------------------------- skeleton -----------------------------
# OpenPose body-18 indices: R-shoulder=2,R-elbow=3,R-wrist=4, L-shoulder=5,L-elbow=6,L-wrist=7
R_ELBOW, L_ELBOW = 3, 6
R_WRIST, L_WRIST = 4, 7
BODY_LIMBS = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),
              (1,11),(11,12),(12,13),(0,14),(0,15),(14,16),(15,17)]
HAND_LIMBS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
              (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
# colors (BGR)
COL_BODY, COL_RIGHT, COL_LEFT, COL_JOINT, COL_INTERP = (255,255,0), (0,255,0), (0,0,255), (0,220,255), (0,255,255)

# ----------------------------- pose IO -----------------------------
def load_dwpose(stem): return pickle.load(open(f"{DWPOSE}/{stem}_dwpose.pkl", "rb"))
def load_wilor(stem):  return pickle.load(open(f"{WILOR}/{stem}_wilor.pkl", "rb"))


def body_points(dwframe, W, H):
    """DWpose best-person body -> list of 18 (x,y,conf) in pixels, or None where the joint is absent."""
    out = [None] * 18
    p = dwframe.get("pose") if isinstance(dwframe, dict) else None
    if not isinstance(p, dict) or p.get("bodies") is None:
        return out
    b = p["bodies"]; cand = np.asarray(b["candidate"]); sub = np.asarray(b["subset"]); sc = np.asarray(b["scores"])
    if sub.shape[0] == 0:
        return out
    for j in range(18):
        idx = int(sub[0, j])
        if idx < 0:
            continue
        x, y = cand[idx] * [W, H]
        out[j] = (float(x), float(y), float(sc[j]) if j < len(sc) else 0.0)
    return out


def best_hands(wlframe, W, H, min_conf=0.4):
    """ONE best (highest-conf) WiLoR hand per side >= min_conf -> {'L':xy21|None, 'R':xy21|None}."""
    out = {"L": None, "R": None}; best = {"L": -1.0, "R": -1.0}
    p = wlframe.get("pose") if isinstance(wlframe, dict) else None
    if not isinstance(p, dict) or p.get("hands") is None:
        return out
    hands = np.asarray(p["hands"]); isr = list(p.get("hands_is_right", [])); cf = list(p.get("hands_conf", []))
    for k in range(len(hands)):
        c = float(cf[k]) if k < len(cf) else 0.0
        side = "R" if (k < len(isr) and isr[k]) else "L"
        if c < min_conf or c <= best[side]:
            continue
        best[side] = c
        out[side] = [(float(hands[k][t][0]*W), float(hands[k][t][1]*H)) for t in range(hands.shape[1])]
    return out


def all_hands(wlframe, W, H, min_conf=0.1):
    """EVERY WiLoR hand >= min_conf -> list of dict(xy=[21x[x,y]], is_right, conf)  (vis_wilor style)."""
    res = []
    p = wlframe.get("pose") if isinstance(wlframe, dict) else None
    if not isinstance(p, dict) or p.get("hands") is None:
        return res
    hands = np.asarray(p["hands"]); isr = list(p.get("hands_is_right", [])); cf = list(p.get("hands_conf", []))
    for k in range(len(hands)):
        c = float(cf[k]) if k < len(cf) else 0.0
        if c < min_conf:
            continue
        res.append(dict(xy=[[float(hands[k][t][0]*W), float(hands[k][t][1]*H)] for t in range(hands.shape[1])],
                        is_right=bool(isr[k]) if k < len(isr) else False, conf=round(c, 3)))
    return res


# ----------------------------- interpolation -----------------------------
def interp_runs(ch, present, view, n):
    """Interpolate a point-channel's interior gaps across the whole clip, within each contiguous VIEW
    run (fills view-1 AND view-2 gaps, never across a camera cut). ch[i]=(x,y)|None, present=bool list.
    Returns (ch, src) where src[i] in {'conf','interp','miss'}."""
    src = ["conf" if present[i] else "miss" for i in range(n)]
    i = 0
    while i < n:
        k = i
        while k < n and view[k] == view[i]:
            k += 1
        valid = [t for t in range(i, k) if present[t]]
        for v0, v1 in zip(valid, valid[1:]):
            if v1 - v0 <= 1:
                continue
            p0, p1 = ch[v0], ch[v1]
            for t in range(v0+1, v1):
                f = (t-v0)/(v1-v0)
                ch[t] = (p0[0]+(p1[0]-p0[0])*f, p0[1]+(p1[1]-p0[1])*f); src[t] = "interp"
        i = k
    return ch, src


def build_interpolated(dw, wl, view, W, H, body_conf=0.4, hand_conf=0.4):
    """Combine DWpose BODY + WiLoR HANDS (one hand per side), interpolate body-elbows and all 21 hand
    kpts per side across the whole clip (per view run). Returns a per-frame list:
      {view, body:[18x[x,y,src]], left_hand:[21x[x,y,src]], right_hand:[21x[x,y,src]]}  (src in conf/interp/miss)."""
    n = len(dw)
    body = [[None]*n for _ in range(18)]; bpres = [[False]*n for _ in range(18)]
    HAND = {"L": [[None]*n for _ in range(21)], "R": [[None]*n for _ in range(21)]}
    HPRES = {"L": [[False]*n for _ in range(21)], "R": [[False]*n for _ in range(21)]}
    for i in range(n):
        for j, bp in enumerate(body_points(dw[i], W, H)):
            if bp is not None and bp[2] >= body_conf:
                body[j][i] = (bp[0], bp[1]); bpres[j][i] = True
        hb = best_hands(wl[i], W, H, hand_conf)
        for side in ("L", "R"):
            if hb[side] is not None:
                for t in range(21):
                    HAND[side][t][i] = hb[side][t]; HPRES[side][t][i] = True
    bsrc = [["conf" if bpres[j][i] else "miss" for i in range(n)] for j in range(18)]
    for j in (R_ELBOW, L_ELBOW):                                  # elbows interpolated
        body[j], bsrc[j] = interp_runs(body[j], bpres[j], view, n)
    hsrc = {"L": [None]*21, "R": [None]*21}
    for side in ("L", "R"):                                       # all hand kpts interpolated
        for t in range(21):
            HAND[side][t], hsrc[side][t] = interp_runs(HAND[side][t], HPRES[side][t], view, n)
    frames = []
    for i in range(n):
        frames.append(dict(view=int(view[i]),
            body=[[round(body[j][i][0],2), round(body[j][i][1],2), bsrc[j][i]] if body[j][i] else [0,0,"miss"] for j in range(18)],
            left_hand=[[round(HAND["L"][t][i][0],2), round(HAND["L"][t][i][1],2), hsrc["L"][t][i]] if HAND["L"][t][i] else [0,0,"miss"] for t in range(21)],
            right_hand=[[round(HAND["R"][t][i][0],2), round(HAND["R"][t][i][1],2), hsrc["R"][t][i]] if HAND["R"][t][i] else [0,0,"miss"] for t in range(21)]))
    return frames


# ----------------------------- views -----------------------------
def load_view_array(stem, n):
    """Per-frame view (1=face / 2=top-down) array of length n, from view_annot/<stem>_views.json."""
    vj = json.load(open(f"{VIEW_ANNOT}/{stem}_views.json"))
    view = np.ones(n, int)
    for s in vj["segments"]:
        view[max(0, s["start"]):min(n, s["end"]+1)] = s["view"]
    return view


def view_segments(view_slice, fps=FPS):
    """contiguous same-view runs -> [{view,start_frame,end_frame,start_s,end_s,n_frames}]."""
    segs = []; n = len(view_slice); i = 0
    while i < n:
        k = i
        while k < n and view_slice[k] == view_slice[i]:
            k += 1
        segs.append(dict(view=int(view_slice[i]), start_frame=i, end_frame=k-1,
                         start_s=round(i/fps, 2), end_s=round((k-1)/fps, 2), n_frames=k-i))
        i = k
    return segs


# ----------------------------- video -----------------------------
def open_writer(path, W, H, fps=FPS, crf=22):
    """ffmpeg rawvideo->mp4 pipe. Feed BGR frames via write_frame(); close_writer() when done."""
    return subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24",
        "-s", f"{W}x{H}", "-r", str(int(round(fps))), "-i", "pipe:0", "-c:v","libx264",
        "-pix_fmt","yuv420p","-crf",str(crf),"-movflags","+faststart", path], stdin=subprocess.PIPE)


def write_frame(ff, img):
    ff.stdin.write(np.ascontiguousarray(img).tobytes())


def close_writer(ff):
    ff.stdin.close(); ff.wait()


def cut_exact(src, a, b, out):
    """Re-encode EXACT source frames a..b (round video frame k == source frame a+k)."""
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i", src,
        "-vf", f"select='between(n,{a},{b})',setpts=PTS-STARTPTS", "-vsync","0",
        "-c:v","libx264","-preset","veryfast","-crf","20","-an","-movflags","+faststart", out], check=True)


def cut_segment(src, start_s, end_s, out):
    """Frame-accurate re-encode of the time window [start_s, end_s]."""
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i", src, "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-c:v","libx264","-preset","veryfast","-crf","20","-an","-movflags","+faststart", out], check=True)


# ----------------------------- rendering -----------------------------
def draw_interp_skeleton(img, F, hand_min_conf=None):
    """Draw a build_interpolated() frame F onto img: body cyan, R-hand green, L-hand red, interp yellow."""
    bd = F["body"]
    for u, v in BODY_LIMBS:
        pu, pv = bd[u], bd[v]
        if pu[2] != "miss" and pv[2] != "miss":
            c = COL_INTERP if "interp" in (pu[2], pv[2]) else COL_BODY
            cv2.line(img, (int(pu[0]),int(pu[1])), (int(pv[0]),int(pv[1])), c, 2)
    for p in bd:
        if p[2] != "miss":
            cv2.circle(img, (int(p[0]),int(p[1])), 4 if p[2]=="interp" else 3,
                       COL_INTERP if p[2]=="interp" else COL_BODY, -1)
    for key, base in (("right_hand", COL_RIGHT), ("left_hand", COL_LEFT)):
        kpts = F[key]
        for u, v in HAND_LIMBS:
            pu, pv = kpts[u], kpts[v]
            if pu[2] != "miss" and pv[2] != "miss":
                c = COL_INTERP if "interp" in (pu[2], pv[2]) else base
                cv2.line(img, (int(pu[0]),int(pu[1])), (int(pv[0]),int(pv[1])), c, 2)
        for p in kpts:
            if p[2] != "miss":
                cv2.circle(img, (int(p[0]),int(p[1])), 3, COL_INTERP if p[2]=="interp" else COL_JOINT, -1)


# ----------------------------- cards -----------------------------
def card_signal(path, roi=(0.25, 0.75, 0.55, 0.95)):
    """Per-frame bright-card area fraction in the central table ROI. Returns (smoothed, fps, N, W, H)."""
    cap = cv2.VideoCapture(path); fps = cap.get(5) or FPS; N = int(cap.get(7)); W = int(cap.get(3)); H = int(cap.get(4))
    x0, x1 = int(roi[0]*W), int(roi[1]*W); y0, y1 = int(roi[2]*H), int(roi[3]*H)
    area = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        area.append(float(cv2.inRange(hsv, (0,0,150), (180,70,255)).mean())/255.0)
    cap.release()
    a = np.array(area)
    sm = np.convolve(a, np.ones(5)/5, mode="same") if len(a) else a
    return sm, fps, N, W, H
