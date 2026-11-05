"""Within-clip DEALER-CHANGE detection via InsightFace (CPU).

Samples the video every --step frames, embeds the dealer's face (largest face, ArcFace), and
segments the clip into dealer identities online: when a run of samples stops matching the current
dealer, a change is declared at the first mismatching frame. Output:

  <out-json>/<stem>_change.json   segments + change times
  <out-vis>/<stem>_change.mp4     the change region(s) (+/- WIN s) with per-frame face box + DEALER label

CPU only (no GPU). Run in the `wilor` env (has insightface). Usage:
  python dealer_change_temporal.py --stem 2025-12-15_20-52-04
"""
import os, json, argparse
import numpy as np, cv2
cv2.setNumThreads(1)

VID = "/home/ubuntu/us-west-3-fs/sahithi/baccarat/vid"
SWITCH = 0.35       # sim to current dealer below this = mismatch
MIN_PERSIST = 12    # a mismatch must persist this many samples (~6s) to count as a real new dealer
COH = 0.50          # ...and those samples must be a coherent identity (not look-down noise)
WIN = 8.0           # seconds of context each side of a change in the viz

DET = (320, 320)    # detector input; dealer face is large so 320 is plenty and ~4x faster than 640
_app = None
def get_app():
    global _app
    if _app is None:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, "4")                 # don't let each worker grab all cores
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'],
                            allowed_modules=['detection', 'recognition'])
        _app.prepare(ctx_id=-1, det_size=DET)
    return _app


def dealer_face(app, img):
    fs = app.get(img)
    if not fs:
        return None
    f = max(fs, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
    return f if f.det_score >= 0.5 else None


def scan(stem, step):
    app = get_app()
    cap = cv2.VideoCapture(f"{VID}/{stem}.mp4"); N = int(cap.get(7)); fps = cap.get(5) or 30
    samples = []; i = 0                                    # (frame, emb); SEQUENTIAL decode, no seeks
    while True:
        ok, im = cap.read()
        if not ok: break
        if i % step == 0:
            f = dealer_face(app, im)
            if f is not None:
                samples.append((i, f.normed_embedding))
        i += 1
    cap.release()

    # identity segmentation: a change is only real if the mismatch PERSISTS (>=MIN_PERSIST samples)
    # and those samples are a COHERENT identity -- a brief look-down (reverts) does NOT count.
    def refmean(lst):
        m = np.mean(lst, axis=0); return m / (np.linalg.norm(m) + 1e-9)
    change_fr = []
    if samples:
        fr = [f for f, _ in samples]; E = [e for _, e in samples]; n = len(E)
        cur = [E[0]]; i = 1
        while i < n:
            if float(refmean(cur) @ E[i]) >= SWITCH:
                cur.append(E[i]); cur = cur[-30:]; i += 1
            else:
                j = i                                       # extent of the mismatch run
                while j < n and float(refmean(cur) @ E[j]) < SWITCH:
                    j += 1
                run = E[i:j]
                rm = refmean(run); coh = float(np.mean([rm @ x for x in run])) if run else 0.0
                if len(run) >= MIN_PERSIST and coh >= COH:  # sustained + coherent => real new dealer
                    change_fr.append(int(fr[i])); cur = run[-30:]
                # else: transient (look-down / noise) -> absorb into current dealer
                i = j
    starts = [0] + change_fr
    ends = [c for c in change_fr] + [N]
    segs = [dict(dealer=d, start_frame=s, end_frame=e-1, start_s=round(s/fps, 2), end_s=round((e-1)/fps, 2))
            for d, (s, e) in enumerate(zip(starts, ends))]
    changes = [dict(at_frame=c, at_s=round(c/fps, 2)) for c in change_fr]
    return dict(clip=stem, n_frames=N, fps=round(fps, 2), n_dealers=len(segs),
                segments=segs, changes=changes, n_samples=len(samples), step=step), fps, N


def seg_of(frame, segs):
    for s in segs:
        if s['start_frame'] <= frame <= s['end_frame']:
            return s['dealer']
    return segs[-1]['dealer'] if segs else 0


def render(stem, rep, fps, out_mp4):
    changes = rep['changes']
    if not changes:
        return 0
    win = int(WIN * fps)
    ranges = [(max(0, c['at_frame']-win), min(rep['n_frames']-1, c['at_frame']+win)) for c in changes]
    cap = cv2.VideoCapture(f"{VID}/{stem}.mp4"); w = int(cap.get(3)); h = int(cap.get(4))
    ff = None; written = 0
    COL = [(0,255,0), (0,128,255), (255,0,255), (0,255,255)]
    chg_frames = {c['at_frame'] for c in changes}
    for a, b in ranges:
        cap.set(1, a)
        for fr in range(a, b+1):
            ok, img = cap.read()
            if not ok: break
            if ff is None:
                import subprocess
                ff = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24",
                    "-s", f"{w}x{h}", "-r", str(int(fps)), "-i", "pipe:0", "-c:v","libx264","-pix_fmt","yuv420p",
                    "-crf","23","-movflags","+faststart", out_mp4], stdin=subprocess.PIPE)
            did = seg_of(fr, rep['segments'])
            col = COL[did % len(COL)]
            cv2.rectangle(img, (0,0), (w,46), (0,0,0), -1)
            cv2.putText(img, f"f{fr} {fr/fps:5.1f}s   DEALER {did}", (12,34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            near = min((abs(fr-cf) for cf in chg_frames), default=999)
            if near < fps*0.8:                              # flash CHANGE for ~0.8s around the switch
                cv2.putText(img, "DEALER CHANGE", (w//2-230, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
            ff.stdin.write(np.ascontiguousarray(img).tobytes()); written += 1
    cap.release()
    if ff: ff.stdin.close(); ff.wait()
    return written


def process_clip(job):
    stem, step, out_json, out_vis, do_viz = job
    try:
        rep, fps, N = scan(stem, step)
        json.dump(rep, open(os.path.join(out_json, f"{stem}_change.json"), "w"), indent=2)
        line = f"  {stem}: dealers={rep['n_dealers']} changes={[c['at_s'] for c in rep['changes']]}s"
        if do_viz and rep['changes']:
            w = render(stem, rep, fps, os.path.join(out_vis, f"{stem}_change.mp4"))
            line += f" + viz({w}f)"
        return line
    except Exception as e:
        return f"  ERR {stem}: {e}"


if __name__ == "__main__":
    import glob, multiprocessing as mp
    B = "/home/ubuntu/us-west-3-fs/sahithi/baccarat"
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=None, help="single clip id (else batch)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=25, help="clips to process in batch mode")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--step", type=int, default=15, help="sample every N frames (~0.5s)")
    ap.add_argument("--out-json", default=f"{B}/dealers/dealer_change_temporal")
    ap.add_argument("--out-vis", default=f"{B}/vis/dealer_change")
    ap.add_argument("--no-viz", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_json, exist_ok=True); os.makedirs(a.out_vis, exist_ok=True)
    allc = sorted(glob.glob(f"{VID}/*.mp4"))
    stems = [a.stem] if a.stem else [os.path.basename(c)[:-4] for c in allc[a.start:a.start + a.limit]]
    jobs = [(s, a.step, a.out_json, a.out_vis, not a.no_viz) for s in stems]
    print(f"dealer-change scan: {len(jobs)} clip(s) step={a.step} procs={a.procs} (CPU)", flush=True)
    if a.procs > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(a.procs) as p:    # spawn: clean onnxruntime per worker
            for m in p.imap_unordered(process_clip, jobs): print(m, flush=True)
    else:
        for j in jobs: print(process_clip(j), flush=True)
    print("DONE", flush=True)
