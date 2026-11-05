"""Regenerate ONLY the interpolated_dwpose_wilor/{pkl} + its viz for the dealer4 rounds.

DWpose body + WiLoR hands (one best hand per side, conf >= HAND_CONF), with body-elbows and all 21
hand kpts per side interpolated across the whole clip (within each view run). See common.build_interpolated.
Run in `wilor` env.  python redo_interp.py
"""
import json, pickle
import cv2
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import common as C

OUT = f"{C.ANNOT}/action_annot_round/dealer4"
S = "2025-12-15_19-53-36"
BODY_CONF = HAND_CONF = 0.4


def render(vid, frames, out):
    cap = cv2.VideoCapture(vid); W = int(cap.get(3)); H = int(cap.get(4))
    ff = C.open_writer(out, W, H); i = 0
    while True:
        ok, img = cap.read()
        if not ok or i >= len(frames):
            break
        F = frames[i]
        C.draw_interp_skeleton(img, F)
        cv2.rectangle(img, (0,0), (W,40), (0,0,0), -1)
        cv2.putText(img, f"view {F['view']}  body(cyan) + wilor hands(R green/L red, conf>={HAND_CONF})  yellow=interp(all views)",
                    (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        C.write_frame(ff, img); i += 1
    cap.release(); C.close_writer(ff); return i


def main():
    dwf = C.load_dwpose(S); Nfull = len(dwf); H, W = dwf[0]["frame_dimensions"]
    view_full = C.load_view_array(S, Nfull)
    for i in range(1, 5):
        rj = json.load(open(f"{OUT}/json/round{i}.json"))
        a, b = rj["orig_start_frame"], rj["orig_end_frame"]
        dw = pickle.load(open(f"{OUT}/dwpose/round{i}_dwpose.pkl", "rb"))
        wl = pickle.load(open(f"{OUT}/wilor/round{i}_wilor.pkl", "rb"))
        frames = C.build_interpolated(dw, wl, view_full[a:b+1], W, H, BODY_CONF, HAND_CONF)
        pickle.dump(dict(source_clip=S, orig_start_frame=a, orig_end_frame=b, fps=C.FPS, n_frames=b-a+1,
                         body_conf=BODY_CONF, hand_min_conf=HAND_CONF,
                         note="body=dwpose; hands=best wilor per side; elbows+hands interpolated per view run (whole clip)",
                         frames=frames),
                    open(f"{OUT}/interpolated_dwpose_wilor/round{i}.pkl", "wb"))
        nf = render(f"{OUT}/vid/round{i}.mp4", frames, f"{OUT}/interpolated_dwpose_wilor_vis/round{i}.mp4")
        ei = sum(1 for f in frames for j in (C.R_ELBOW, C.L_ELBOW) if f["body"][j][2] == "interp")
        hp = lambda side: sum(1 for f in frames if any(p[2] != "miss" for p in f[side]))
        print(f"round{i}: {nf}f  elbow-interp={ei}  L-hand={hp('left_hand')}  R-hand={hp('right_hand')}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
