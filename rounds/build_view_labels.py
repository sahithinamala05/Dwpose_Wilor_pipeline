"""Add per-round VIEW-1/VIEW-2 labeling to the dealer4 round artifact tree.

For each round, slice the clip's view segmentation to the round's frame range, re-base to 0, and write:
  view_label/round{i}.json        {segments:[{view,start_frame,end_frame,start_s,end_s}], per_frame_view:[...]}
  view_label_vis/round{i}.mp4     round video + VIEW-1/VIEW-2 label overlay + timeline bar
Run in `wilor` env.  python build_view_labels.py
"""
import os, json
import cv2
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import common as C

OUT = f"{C.ANNOT}/action_annot_round/dealer4"
S = "2025-12-15_19-53-36"
COL = {1: (0, 200, 255), 2: (255, 160, 0)}      # view1 amber / view2 blue (BGR)
NAME = {1: "VIEW 1 (face)", 2: "VIEW 2 (top-down)"}


def render(vid, view_slice, segs, out):
    cap = cv2.VideoCapture(vid); W = int(cap.get(3)); H = int(cap.get(4)); N = int(cap.get(7))
    ff = C.open_writer(out, W, H); by = H-34; i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        v = int(view_slice[i]) if i < len(view_slice) else 1
        cv2.rectangle(img, (0,0), (W,44), (0,0,0), -1)
        cv2.putText(img, f"{i/C.FPS:5.1f}s   {NAME[v]}", (12,32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL[v], 3)
        cv2.rectangle(img, (0,by), (W,by+34), (20,20,20), -1)
        for s in segs:
            x0 = int(s["start_frame"]/max(1,N)*W); x1 = int((s["end_frame"]+1)/max(1,N)*W)
            cv2.rectangle(img, (x0,by+6), (x1,by+28), COL[s["view"]], -1)
        cx = int(i/max(1,N)*W); cv2.line(img, (cx,by), (cx,by+34), (255,255,255), 2)
        C.write_frame(ff, img); i += 1
    cap.release(); C.close_writer(ff); return i


def main():
    os.makedirs(f"{OUT}/view_label", exist_ok=True); os.makedirs(f"{OUT}/view_label_vis", exist_ok=True)
    view_full = C.load_view_array(S, len(C.load_dwpose(S)))
    for i in range(1, 5):
        rj = json.load(open(f"{OUT}/json/round{i}.json"))
        a, b = rj["orig_start_frame"], rj["orig_end_frame"]
        vs = view_full[a:b+1]; segs = C.view_segments(vs)
        n1 = int((vs == 1).sum()); n2 = int((vs == 2).sum())
        json.dump(dict(source_clip=S, orig_start_frame=a, orig_end_frame=b, fps=C.FPS, n_frames=len(vs),
                       n_view1=n1, n_view2=n2, segments=segs, per_frame_view=[int(x) for x in vs]),
                  open(f"{OUT}/view_label/round{i}.json", "w"), indent=1)
        nf = render(f"{OUT}/vid/round{i}.mp4", vs, segs, f"{OUT}/view_label_vis/round{i}.mp4")
        print(f"round{i}: {len(vs)}f  view1={n1} view2={n2}  segments={len(segs)}  vis={nf}f", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
