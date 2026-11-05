"""Temporal segmentation by FACE VISIBILITY, from DWpose pkls.

For every frame we measure how much of the dealer's face DWpose sees (fraction of the 68 face
keypoints whose confidence >= --conf). A frame is:
    VIEW 1  if the face is visible  (visible fraction >= --min-frac)
    VIEW 2  if the face is NOT visible (dealer turned away / looking down / no detection)
Contiguous frames of the same view form a segment. Short runs (< --min-run frames) are merged
into their neighbour to avoid flicker. Output: per-clip JSON of {view, start, end, n_frames, t_*}.

Reads STAGE-1 DWpose pkls (<stem>_dwpose.pkl) -- they carry 'faces' + 'face_scores'.
(The WiLoR pkls are hands-only, so they are NOT used here.) Run in the `wilor` conda env.

Output: one <clip>_views.json per clip into --out (default baccarat/temporal_annotation/).
Usage:
  python face_view_segment.py --pkl /path/<stem>_dwpose.pkl        # one clip
  python face_view_segment.py --dir baccarat/dwpose               # batch a folder
  python face_view_segment.py --pkl ... --viz --src baccarat/vid  # + labelled video
Options: --out DIR  --conf 0.3  --min-frac 0.5  --min-run 5  --fps 30
"""
import os, sys, glob, json, pickle, argparse
import numpy as np

VIEW_VISIBLE, VIEW_HIDDEN = 1, 2    # view 1 = face visible, view 2 = face not visible


def visible_fraction(frame, conf):
    """Fraction (0..1) of the 68 face keypoints visible for the best-detected person.
    Uses face_scores only (no coord fallback). 0.0 means no face/pose detected -> VIEW 2."""
    p = frame.get('pose') if isinstance(frame, dict) else None
    if not isinstance(p, dict):
        return 0.0
    fs = p.get('face_scores')
    if fs is None:
        return 0.0
    fs = np.asarray(fs)
    if fs.ndim != 2 or fs.shape[0] == 0:
        return 0.0
    i = int(np.argmax(fs.mean(axis=1)))                # the person whose face is most confident (the dealer)
    return float(np.mean(fs[i] >= conf))


def runs_of(labels):
    """[v0,v0,v1,...] -> list of [start, end_inclusive, value]."""
    runs, s = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            runs.append([s, i - 1, labels[s]]); s = i
    return runs


def enforce_min_run(labels, min_run):
    """Merge any run shorter than min_run into the (longer) adjacent run, until none remain."""
    labels = list(labels)
    while True:
        runs = runs_of(labels)
        short = [r for r in runs if (r[1] - r[0] + 1) < min_run]
        if not short or len(runs) == 1:
            break
        r = min(short, key=lambda x: x[1] - x[0])       # shortest offending run
        i = runs.index(r)
        left = runs[i - 1] if i > 0 else None
        right = runs[i + 1] if i < len(runs) - 1 else None
        if left and right:
            newv = left[2] if (left[1] - left[0]) >= (right[1] - right[0]) else right[2]
        else:
            newv = (left or right)[2]
        for t in range(r[0], r[1] + 1):
            labels[t] = newv
    return labels


def segment_clip(frames, conf, min_frac, min_run, fps):
    fracs = np.array([visible_fraction(fr, conf) for fr in frames])
    labels = np.where(fracs >= min_frac, VIEW_VISIBLE, VIEW_HIDDEN)
    labels = enforce_min_run(labels, min_run)
    segs = []
    for s, e, v in runs_of(labels):
        segs.append(dict(view=int(v), start=int(s), end=int(e), n_frames=int(e - s + 1),
                         t_start=round(s / fps, 3), t_end=round((e + 1) / fps, 3)))
    n1 = int(sum(s['n_frames'] for s in segs if s['view'] == VIEW_VISIBLE))
    n2 = len(frames) - n1
    summary = dict(view1_visible_frames=n1, view2_hidden_frames=n2, n_segments=len(segs),
                   view1_pct=round(100 * n1 / max(1, len(frames)), 1))
    return labels, segs, summary


def process_pkl(pkl, conf, min_frac, min_run, fps):
    frames = pickle.load(open(pkl, 'rb'))
    labels, segs, summary = segment_clip(frames, conf, min_frac, min_run, fps)
    stem = os.path.basename(pkl)
    for suf in ('_dwpose.pkl', '.pkl'):
        if stem.endswith(suf): stem = stem[:-len(suf)]; break
    return dict(clip=stem, n_frames=len(frames), fps=fps,
                params=dict(conf=conf, min_frac=min_frac, min_run=min_run),
                segments=segs, summary=summary), labels


def render_viz(pkl, video, segs, labels, out_mp4, fps):
    import cv2, subprocess
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video); w = int(cap.get(3)); h = int(cap.get(4))
    if w <= 0 or h <= 0:
        cap.release(); raise RuntimeError(f"unreadable video dims ({w}x{h}) for {video}")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(int(fps)), "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-movflags", "+faststart", out_mp4], stdin=subprocess.PIPE)
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        v = int(labels[idx]) if idx < len(labels) else VIEW_HIDDEN
        col = (0, 200, 0) if v == VIEW_VISIBLE else (0, 0, 255)
        txt = "VIEW 1  face visible" if v == VIEW_VISIBLE else "VIEW 2  face hidden"
        cv2.rectangle(img, (0, 0), (w, 54), (0, 0, 0), -1)
        cv2.putText(img, f"f{idx}  {txt}", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
        ff.stdin.write(np.ascontiguousarray(img).tobytes()); idx += 1
    cap.release(); ff.stdin.close(); ff.wait()


def _run_one(job):
    pk, conf, min_frac, min_run, fps, out_dir, viz, viz_out, src = job
    try:
        res, labels = process_pkl(pk, conf, min_frac, min_run, fps)
        cid = res["clip"]; s = res["summary"]
        json.dump(res, open(os.path.join(out_dir, f"{cid}_views.json"), "w"), indent=2)
        line = (f"  {cid}: {s['n_segments']} seg | view1 {s['view1_pct']}% "
                f"({s['view1_visible_frames']}f) view2 ({s['view2_hidden_frames']}f) -> json")
        if viz:
            vp = next((os.path.join(src, cid + e) for e in (".mp4", ".mkv")
                       if os.path.exists(os.path.join(src, cid + e))), None)
            if vp:
                render_viz(pk, vp, res["segments"], labels, os.path.join(viz_out, f"{cid}_views.mp4"), fps)
                line += " + viz"
            else:
                line += " (no video; skipped viz)"
        return line
    except Exception as e:
        return f"  ERR {os.path.basename(pk)}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", help="single DWpose pkl")
    ap.add_argument("--dir", help="folder of *_dwpose.pkl to batch")
    ap.add_argument("--out", default="/home/ubuntu/us-west-3-fs/sahithi/baccarat/temporal_annotation",
                    help="output folder for the <clip>_views.json per clip")
    ap.add_argument("--viz-out", default="/home/ubuntu/us-west-3-fs/sahithi/baccarat/vis/temporal_vis",
                    help="output folder for the <clip>_views.mp4 with --viz")
    ap.add_argument("--conf", type=float, default=0.3, help="per-keypoint confidence to count as visible")
    ap.add_argument("--min-frac", type=float, default=0.5, help="fraction of 68 face kpts visible -> VIEW 1")
    ap.add_argument("--min-run", type=int, default=5, help="merge segments shorter than this many frames")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate (for t_start/t_end seconds)")
    ap.add_argument("--limit", type=int, default=0, help="cap clips in --dir mode")
    ap.add_argument("--viz", action="store_true", help="also render a view-labelled video (needs --src)")
    ap.add_argument("--src", default="/home/ubuntu/us-west-3-fs/sahithi/baccarat/vid", help="video dir for --viz")
    ap.add_argument("--procs", type=int, default=1, help="parallel clips (viz rendering is the slow part)")
    a = ap.parse_args()

    if a.pkl:
        pkls = [a.pkl]
    elif a.dir:
        pkls = sorted(glob.glob(os.path.join(a.dir, "*_dwpose.pkl")))
        if a.limit: pkls = pkls[:a.limit]
    else:
        ap.error("pass --pkl or --dir")

    os.makedirs(a.out, exist_ok=True)
    if a.viz: os.makedirs(a.viz_out, exist_ok=True)
    jobs = [(pk, a.conf, a.min_frac, a.min_run, a.fps, a.out, a.viz, a.viz_out, a.src) for pk in pkls]
    print(f"segmenting {len(jobs)} clip(s)  procs={a.procs} viz={a.viz} -> {a.out}"
          + (f" (+ viz {a.viz_out})" if a.viz else ""), flush=True)
    if a.procs > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(a.procs) as p:
            for m in p.imap_unordered(_run_one, jobs): print(m, flush=True)
    else:
        for j in jobs: print(_run_one(j), flush=True)
    print("DONE", flush=True)
