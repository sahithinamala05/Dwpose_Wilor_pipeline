"""Segment baccarat clips into individual ROUNDS via the table card-presence signal, and cut each
round into its own clip for annotation.

A round = a card-present interval (cards dealt -> cards cleared) padded by PRE/POST seconds so the clip
contains the full cycle (wait -> deal -> reveal -> discard). Cuts are frame-accurate re-encodes; a
sidecar manifest records each round's offset back into the source video.

  <out>/<dealer>/<stem>_r<idx>.mp4           round clips
  <out>/<dealer>/_rounds_manifest.json       [{file, src, orig_start_s, orig_end_s, dur_s}]

Usage (probe one clip):   python segment_rounds.py --probe /path/clip.mp4
       (segment a dealer): python segment_rounds.py --dealer dealer_4
"""
import os, json, glob, argparse
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import common as C

SRC_ROOT = f"{C.DEALERS}/vid_single_dealer"
PRE, POST = 6.0, 4.0          # seconds of context before deal / after discard
MERGE_GAP = 3.0               # merge card-present blips closer than this (same round)
MIN_PRESENT = 1.0            # ignore card-present blips shorter than this


def find_rounds(path):
    ac, fps, N, _, _ = C.card_signal(path)
    if len(ac) == 0:
        return [], fps, N
    thr = max(0.015, ac.max()*0.25)
    present = (ac > thr).astype(int)
    ivals = []; s = None
    for j in range(len(present)):
        if present[j] and s is None: s = j
        if not present[j] and s is not None: ivals.append([s, j-1]); s = None
    if s is not None: ivals.append([s, len(present)-1])
    ivals = [iv for iv in ivals if (iv[1]-iv[0]) >= fps*MIN_PRESENT]
    # merge near intervals (same round, brief area dip)
    merged = []
    for iv in ivals:
        if merged and (iv[0]-merged[-1][1]) <= fps*MERGE_GAP:
            merged[-1][1] = iv[1]
        else:
            merged.append(iv[:])
    # Each card-present interval ends at a DISCARD. One round = discard-to-discard (wait -> deal ->
    # reveal(combo visible) -> discard). Cut between consecutive clear points.
    clears = [b0 for _, b0 in merged]           # frame of each discard
    rounds = []; prev = 0
    for c in clears:
        s = prev; e = min(N-1, c + POST*fps)
        if (e - s) >= fps*8:                     # skip too-short fragments
            rounds.append(dict(orig_start_s=round(s/fps, 2), orig_end_s=round(e/fps, 2),
                               dur_s=round((e-s)/fps, 2), discard_s=round(c/fps, 2)))
        prev = c + 1                              # next round starts right after this discard
    # trailing round after the last discard (deal+reveal, discard may be in next source clip)
    if (N-1 - prev) >= fps*12:
        rounds.append(dict(orig_start_s=round(prev/fps, 2), orig_end_s=round((N-1)/fps, 2),
                           dur_s=round((N-1-prev)/fps, 2), discard_s=None))
    return rounds, fps, N


def segment_dealer(dealer, out_root):
    srcdir = f"{SRC_ROOT}/{dealer}"
    vids = sorted(glob.glob(f"{srcdir}/*.mp4"))
    outdir = f"{out_root}/{dealer}"; os.makedirs(outdir, exist_ok=True)
    manifest = []; ri = 0
    for v in vids:
        stem = os.path.basename(v)[:-4]
        rounds, fps, N = find_rounds(v)
        print(f"  {stem}: {len(rounds)} rounds", flush=True)
        for k, r in enumerate(rounds):
            out = f"{outdir}/{stem}_r{k}.mp4"
            try:
                if not os.path.exists(out):
                    C.cut_segment(os.path.realpath(v), r["orig_start_s"], r["orig_end_s"], out)
            except Exception as e:
                print(f"    cut ERR {out}: {e}", flush=True); continue
            manifest.append(dict(file=os.path.basename(out), src=stem, **r)); ri += 1
    json.dump(manifest, open(f"{outdir}/_rounds_manifest.json", "w"), indent=2)
    print(f"DONE {dealer}: {ri} round clips -> {outdir}", flush=True)
    return ri


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=None, help="print detected rounds for one clip (no cutting)")
    ap.add_argument("--dealer", default=None, help="segment a dealer_<k> folder")
    ap.add_argument("--out", default=f"{C.ANNOT}/annotate_batch")
    a = ap.parse_args()
    if a.probe:
        rounds, fps, N = find_rounds(a.probe)
        print(f"{os.path.basename(a.probe)}: {len(rounds)} rounds ({round(N/fps,1)}s total)")
        for i, r in enumerate(rounds):
            print(f"  r{i}: {r['orig_start_s']}-{r['orig_end_s']}s ({r['dur_s']}s)  discard@{r['discard_s']}")
    elif a.dealer:
        segment_dealer(a.dealer, a.out)
    else:
        print("specify --probe <clip> or --dealer dealer_<k>")
