"""Cut View-2 (top-down) and stitch all View-1 (face-visible) segments into ONE continuous video.

Removes the camera switches so a VLM sees a single uninterrupted dealer-facing stream. Writes:
  <out-vid>/<stem>_view1.mp4      concatenated View-1 frames
  <out-map>/<stem>_view1_map.json  stitched_frame -> original_frame (+ fps, segments kept)

Run in the `wilor` env (ffmpeg + cv2). Usage:
  python stitch_view1.py --stem 2025-12-15_19-20-48
"""
import os, json, argparse
import cv2
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import common as C


def stitch(stem, out_vid, out_map):
    vj = json.load(open(f"{C.VIEW_ANNOT}/{stem}_views.json"))
    fps = vj.get("fps", C.FPS) or C.FPS
    segs = [s for s in vj["segments"] if s["view"] == 1]        # keep face-visible only
    keep_set = set()
    for s in segs:
        keep_set.update(range(s["start"], s["end"] + 1))

    cap = cv2.VideoCapture(f"{C.VID}/{stem}.mp4")
    w = int(cap.get(3)); h = int(cap.get(4)); N = int(cap.get(7))
    out = os.path.join(out_vid, f"{stem}_view1.mp4")
    ff = C.open_writer(out, w, h, fps, crf=20)

    stitched_to_orig = []                                       # SEQUENTIAL decode, no seeks
    i = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        if i in keep_set:
            C.write_frame(ff, img)
            stitched_to_orig.append(i)
        i += 1
    cap.release(); C.close_writer(ff)

    rep = dict(clip=stem, fps=round(float(fps), 3), orig_frames=N,
               view1_frames=len(stitched_to_orig),
               view1_duration_s=round(len(stitched_to_orig) / fps, 2),
               segments_kept=[dict(start=s["start"], end=s["end"]) for s in segs],
               stitched_to_orig=stitched_to_orig)
    json.dump(rep, open(os.path.join(out_map, f"{stem}_view1_map.json"), "w"))
    return out, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out-vid", default=f"{C.B}/view1_stitched")
    ap.add_argument("--out-map", default=f"{C.B}/view1_stitched")
    a = ap.parse_args()
    os.makedirs(a.out_vid, exist_ok=True); os.makedirs(a.out_map, exist_ok=True)
    out, rep = stitch(a.stem, a.out_vid, a.out_map)
    print(f"{a.stem}: view1 {rep['view1_frames']}f ({rep['view1_duration_s']}s) "
          f"from {rep['orig_frames']}f, {len(rep['segments_kept'])} segs -> {out}", flush=True)
