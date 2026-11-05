"""Segregate baccarat footage by DEALER IDENTITY into vid_single_dealer/dealer_<k>/ (incremental).

For each source clip (processed in time order):
  1. ffmpeg decodes it subsampled to SAMPLE_FPS frames (scaled), InsightFace-GPU embeds the dealer
     (largest face) per sample.
  2. Online within-clip segmentation splits the clip into per-dealer runs (a sustained, coherent face
     change = a real dealer change; brief look-downs are absorbed).
  3. Each run is matched ONLINE to a global dealer identity (greedy cosine threshold); new identities
     open a new dealer_<k> folder. The run is emitted IMMEDIATELY:
       - whole-clip single-dealer run  -> symlink  dealer_<k>/<stem>.mp4
       - a cut run from a changed clip  -> frame-accurate re-encode dealer_<k>/<stem>_d<i>.mp4
So dealer_<k>/ folders populate as the scan runs (no end-of-run barrier).

Outputs: vid_single_dealer/dealer_<k>/... , single_dealer_scan/<stem>.json , single_dealer_scan/_dealers.json
InsightFace on GPU (shares the GPU with DWpose). Run in `wilor` env with torch's cuDNN on LD_LIBRARY_PATH:
  LD_LIBRARY_PATH=".../torch/lib:$LD_LIBRARY_PATH" python single_dealer_filter.py --start 0 --limit 300
"""
import os, json, glob, argparse, subprocess
import numpy as np, cv2; cv2.setNumThreads(1)

B = "/home/ubuntu/us-west-3-fs/sahithi/baccarat"
VID = f"{B}/vid"
SAMPLE_FPS = 1.0        # face samples per second
SCALE = (960, 540)      # detection-input frame size (dealer face stays large)
SWITCH = 0.35           # sim-to-current-dealer below this = mismatch
MIN_PERSIST_S = 6.0     # a mismatch must persist this long to count as a real new dealer
COH = 0.50              # ...and those samples must be a coherent identity (not look-down noise)
MIN_FACE_FRAC = 0.5     # need a face in >=50% of samples to trust the identity call
MIN_SEG_S = 5.0         # don't emit runs shorter than this

_APP = None
def get_app_gpu(det=320):
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis
        _APP = FaceAnalysis(name='buffalo_l',
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                            allowed_modules=['detection', 'recognition'])
        _APP.prepare(ctx_id=0, det_size=(det, det))
    return _APP


def dealer_face(app, img):
    fs = app.get(img)
    if not fs:
        return None
    f = max(fs, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
    return f if f.det_score >= 0.5 else None


def sample_embeddings(stem):
    """ffmpeg subsample -> list of (orig_frame, normed_emb). Returns (samples, fps, N)."""
    cap = cv2.VideoCapture(f"{VID}/{stem}.mp4"); fps = cap.get(5) or 30; N = int(cap.get(7)); cap.release()
    W, H = SCALE
    cmd = ["ffmpeg", "-v", "error", "-i", f"{VID}/{stem}.mp4",
           "-vf", f"fps={SAMPLE_FPS},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:"]
    out = subprocess.run(cmd, capture_output=True).stdout
    nf = len(out) // (W*H*3)
    if nf == 0:
        return [], fps, N
    frames = np.frombuffer(out[:nf*W*H*3], np.uint8).reshape(nf, H, W, 3)
    app = get_app_gpu()
    samples = []
    for k in range(nf):
        f = dealer_face(app, frames[k])
        if f is not None:
            orig_frame = int(round((k / SAMPLE_FPS) * fps))
            samples.append((min(orig_frame, N-1), f.normed_embedding))
    return samples, fps, N


def segment_runs(samples, fps, N):
    """online identity segmentation -> runs with mean embedding (same rule as dealer_change_temporal)."""
    def refmean(lst):
        m = np.mean(lst, axis=0); return m / (np.linalg.norm(m) + 1e-9)
    fr = [f for f, _ in samples]; E = [e for _, e in samples]; n = len(E)
    persist = max(3, int(round(MIN_PERSIST_S * SAMPLE_FPS)))
    change_fr = []; cur = [E[0]]; i = 1
    while i < n:
        if float(refmean(cur) @ E[i]) >= SWITCH:
            cur.append(E[i]); cur = cur[-30:]; i += 1
        else:
            j = i
            while j < n and float(refmean(cur) @ E[j]) < SWITCH:
                j += 1
            run = E[i:j]; rm = refmean(run); coh = float(np.mean([rm @ x for x in run])) if run else 0.0
            if len(run) >= persist and coh >= COH:
                change_fr.append(int(fr[i])); cur = run[-30:]
            i = j
    starts = [0] + change_fr; ends = change_fr + [N]
    runs = []
    for d, (s, e) in enumerate(zip(starts, ends)):
        idx = [t for t, f in enumerate(fr) if s <= f < e]
        if not idx:
            continue
        emb = refmean([E[t] for t in idx])
        runs.append(dict(dealer_local=d, start_frame=s, end_frame=e-1,
                         start_s=round(s/fps, 2), end_s=round((e-1)/fps, 2),
                         n=len(idx), emb=emb.astype(np.float32)))
    return runs, (len(change_fr) == 0)


def cut_segment(src, start_s, end_s, out_mp4):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-an", "-movflags", "+faststart", out_mp4], check=True)


class OnlineClusters:
    """greedy global face clustering; dealer ids are 1-based by first appearance."""
    def __init__(self, sim): self.sim = sim; self.cents = []; self.counts = []
    def assign(self, v):
        v = v / (np.linalg.norm(v) + 1e-9)
        best, bs = -1, self.sim
        for c, cen in enumerate(self.cents):
            s = float(cen @ v)
            if s >= bs: bs = s; best = c
        if best < 0:
            self.cents.append(v.copy()); self.counts.append(1); return len(self.cents)   # new dealer id
        n = self.counts[best]; self.cents[best] = (self.cents[best]*n + v)
        self.cents[best] /= np.linalg.norm(self.cents[best]) + 1e-9; self.counts[best] = n+1
        return best + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--sim", type=float, default=0.45, help="cosine sim to merge runs into one dealer")
    ap.add_argument("--out-dir", default=f"{B}/dealers/vid_single_dealer")
    ap.add_argument("--json-out", default=f"{B}/dealers/single_dealer_scan")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.json_out, exist_ok=True)

    allc = sorted(glob.glob(f"{VID}/*.mp4"))[a.start:a.start + a.limit]
    stems = [os.path.basename(c)[:-4] for c in allc]
    print(f"identity scan (incremental): {len(stems)} clips, {SAMPLE_FPS}fps samples, InsightFace GPU", flush=True)

    clusters = OnlineClusters(a.sim)
    dealers = {}; counts = dict(single=0, multi=0, no_face=0)
    for done, stem in enumerate(stems, 1):
        try:
            samples, fps, N = sample_embeddings(stem)
        except Exception as e:
            print(f"[{done}/{len(stems)}] {stem}: ERROR {e}", flush=True); continue
        expected = max(1, int(N / fps * SAMPLE_FPS))
        face_frac = len(samples) / expected
        if face_frac < MIN_FACE_FRAC or len(samples) < 5:
            counts["no_face"] += 1
            json.dump(dict(clip=stem, status="no_face", face_frac=round(face_frac, 2)),
                      open(f"{a.json_out}/{stem}.json", "w"), indent=2)
            print(f"[{done}/{len(stems)}] {stem}: no-face (frac={face_frac:.2f})", flush=True); continue
        runs, whole = segment_runs(samples, fps, N)
        status = "single" if whole else "multi"; counts[status] += 1
        src = f"{VID}/{stem}.mp4"; out_runs = []
        for r in runs:
            if (r["end_s"] - r["start_s"]) < MIN_SEG_S:
                continue
            k = clusters.assign(r["emb"]); folder = f"{a.out_dir}/dealer_{k}"; os.makedirs(folder, exist_ok=True)
            if whole:
                out = f"{folder}/{stem}.mp4"
                if not os.path.lexists(out): os.symlink(src, out)
            else:
                out = f"{folder}/{stem}_d{r['dealer_local']}.mp4"
                if not os.path.exists(out):
                    try: cut_segment(src, r["start_s"], r["end_s"] + 0.001, out)
                    except Exception as e: print(f"    cut ERR {out}: {e}", flush=True); continue
            d = dealers.setdefault(k, dict(id=k, n_videos=0, seconds=0.0, clips=[]))
            d["n_videos"] += 1; d["seconds"] = round(d["seconds"] + (r["end_s"]-r["start_s"]), 1)
            d["clips"].append(os.path.basename(out))
            out_runs.append(dict(dealer_id=k, file=os.path.basename(out),
                                 start_s=r["start_s"], end_s=r["end_s"],
                                 start_frame=r["start_frame"], end_frame=r["end_frame"]))
        json.dump(dict(clip=stem, status=status, fps=round(float(fps), 2), n_frames=N,
                       face_frac=round(face_frac, 2), n_dealers=len(runs), runs=out_runs),
                  open(f"{a.json_out}/{stem}.json", "w"), indent=2)
        # refresh live summary each clip
        json.dump(dict(scanned=done, total=len(stems), counts=counts, sim=a.sim,
                       n_dealers=len(dealers),
                       dealers={f"dealer_{k}": dealers[k] for k in sorted(dealers)}),
                  open(f"{a.json_out}/_dealers.json", "w"), indent=2)
        print(f"[{done}/{len(stems)}] {stem}: {status} runs={len(out_runs)} "
              f"dealers_so_far={len(dealers)} (ids={sorted(set(r['dealer_id'] for r in out_runs))})", flush=True)

    print("\n==== DONE ====")
    print(f"  clips: single={counts['single']} multi={counts['multi']} no_face={counts['no_face']}")
    print(f"  distinct dealers: {len(dealers)}")
    for k in sorted(dealers):
        print(f"  dealer_{k}: {dealers[k]['n_videos']} videos, {dealers[k]['seconds']}s")
    print(f"  -> {a.out_dir}/dealer_*   summary -> {a.json_out}/_dealers.json")


if __name__ == "__main__":
    main()
