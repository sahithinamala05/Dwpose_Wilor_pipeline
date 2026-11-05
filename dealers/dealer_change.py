"""Scan consecutive clips (chronological) and find where the DEALER changes, via face identity.

Per clip, sample --frames View-1-ish frames, detect the dealer's face (largest face) with
InsightFace (CPU), and average the embeddings into one per-clip identity vector. Then compare
each clip to the previous: a low cosine similarity = a new dealer came in between those clips.
(Clips are 166 s segments named by timestamp, so sorting = time order.)

CPU only (providers=CPUExecutionProvider) so it doesn't touch the GPU inference.

Output: <out>/dealer_scan.json  (per-clip embedding + sim-to-prev + change flags) and a printed list
of change boundaries. Run in the `wilor` conda env (has insightface).

Usage: python dealer_change.py --n-clips 60 --frames 10 --procs 6 --sim-thresh 0.4
"""
import os, glob, json, argparse
import numpy as np, cv2
cv2.setNumThreads(1)

VID = "/home/ubuntu/us-west-3-fs/sahithi/baccarat/vid"
_app = None


def _init():
    global _app
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from insightface.app import FaceAnalysis
    _app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'],
                        allowed_modules=['detection', 'recognition'])
    _app.prepare(ctx_id=-1, det_size=(640, 640))


def clip_embedding(args):
    cid, vid, K = args
    cap = cv2.VideoCapture(vid)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        cap.release(); return cid, None, 0
    embs = []
    for t in np.linspace(0, n - 1, K).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, img = cap.read()
        if not ok:
            continue
        faces = _app.get(img)
        if not faces:
            continue
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))  # dealer = largest face
        embs.append(f.normed_embedding)
    cap.release()
    if not embs:
        return cid, None, 0
    e = np.mean(embs, axis=0); e = e / (np.linalg.norm(e) + 1e-9)
    return cid, e.astype(np.float32).tolist(), len(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ubuntu/us-west-3-fs/sahithi/baccarat/dealers/dealer_change")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-clips", type=int, default=60)
    ap.add_argument("--frames", type=int, default=10, help="frames sampled per clip")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--sim-thresh", type=float, default=0.40, help="cosine sim below this = dealer change")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    clips = sorted(glob.glob(os.path.join(VID, "*.mp4")))[a.start:a.start + a.n_clips]
    jobs = [(os.path.basename(c)[:-4], c, a.frames) for c in clips]
    print(f"scanning {len(jobs)} clips x {a.frames} frames (CPU, procs={a.procs})...", flush=True)

    from multiprocessing import Pool
    results = {}
    with Pool(a.procs, initializer=_init) as p:
        for cid, emb, nf in p.imap(clip_embedding, jobs):      # imap preserves order
            results[cid] = (emb, nf)

    # consecutive cosine similarity + change flags
    order = [os.path.basename(c)[:-4] for c in clips]
    rows, changes, prev = [], [], None
    for cid in order:
        emb, nf = results[cid]
        sim = None
        if emb is not None and prev is not None:
            sim = float(np.dot(np.array(prev), np.array(emb)))
        change = (sim is not None and sim < a.sim_thresh)
        rows.append(dict(clip=cid, n_faces=nf, sim_to_prev=round(sim, 3) if sim is not None else None, change=change))
        if change:
            changes.append((cid, round(sim, 3)))
        if emb is not None:
            prev = emb
    json.dump(dict(sim_thresh=a.sim_thresh, frames=a.frames, rows=rows,
                   embeddings={cid: results[cid][0] for cid in order}),
              open(os.path.join(a.out, "dealer_scan.json"), "w"))

    print("\n=== clip | faces | sim-to-prev | CHANGE ===", flush=True)
    for r in rows:
        mark = "  <<< DEALER CHANGE" if r["change"] else ("" if r["n_faces"] else "  (no face)")
        print(f"  {r['clip']}  faces={r['n_faces']:<2}  sim={r['sim_to_prev']}{mark}", flush=True)
    print(f"\nDETECTED {len(changes)} change boundary(ies) at sim<{a.sim_thresh}:", flush=True)
    for cid, sim in changes:
        print(f"  -> dealer change around {cid} (sim {sim})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
