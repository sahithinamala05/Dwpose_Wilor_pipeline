"""Build the per-round artifact tree for dealer_4's annotated clip (2025-12-15_19-53-36).

Per round (derived from the manual action labels), under annotation/action_annot_round/dealer4/:
  json/round{i}.json                         action timeline, frames re-based to 0 (background fills gaps)
  vid/round{i}.mp4                            round video, EXACT frames a..b (aligns 1:1 with the pkls)
  vis_round/round{i}.mp4                      round video + action-label overlay
  dwpose/round{i}_dwpose.pkl                  sliced DWpose
  wilor/round{i}_wilor.pkl                    sliced WiLoR
  interpolated_dwpose_wilor/round{i}.pkl      DWpose body + WiLoR hands, elbows+hands interpolated (common.build_interpolated)
  interpolated_dwpose_wilor_vis/round{i}.mp4  the interpolated skeleton on the round video
(View-1/2 labeling is added separately by build_view_labels.py.)

Run in the `wilor` env.  python build_round_artifacts.py
"""
import os, json, pickle
import cv2
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
import common as C

S = "2025-12-15_19-53-36"
OUT = f"{C.ANNOT}/action_annot_round/dealer4"
BODY_CONF = HAND_CONF = 0.4
ACTION_COL = {"wait_for_bets":(255,170,0),"close_bets":(0,208,160),"initial_hands":(34,204,34),
              "banker_hit":(0,140,255),"player_hit":(255,80,192),"announce":(0,192,224),
              "discard":(64,64,255),"background":(90,90,90)}


def rounds_from_ann():
    """split the manual action labels into rounds (one per wait_for_bets..discard cycle)."""
    d = json.load(open(f"{C.ANNOT}/action_annotations_manual/dealer_4/{S}_predictions.json"))
    acts = [s for s in d["timeline_segments"] if s["labels"] != ["background"]]
    rounds = []; cur = []
    for s in acts:
        cur.append(s)
        if s["labels"][0] == "discard":
            rounds.append(cur); cur = []
    if cur:
        rounds.append(cur)
    return [(r[0]["start_frame"], r[-1]["end_frame"], r) for r in rounds]


def round_json(r_segs, a, b, path):
    """round-local action timeline (frames re-based to 0, background filling gaps)."""
    n = b - a + 1
    def seg(s, e, lab, meta):
        return dict(start_frame=s, end_frame=e, labels=lab, meta_text=meta,
                    duration_frames=e-s+1, id="", bounding_boxes=[])
    tl = []; cur = 0
    for s in r_segs:
        ss, ee = s["start_frame"]-a, s["end_frame"]-a
        if ss > cur:
            tl.append(seg(cur, ss-1, ["background"], []))
        tl.append(seg(ss, ee, s["labels"], s.get("meta_text") or []))
        cur = ee + 1
    if cur <= n-1:
        tl.append(seg(cur, n-1, ["background"], []))
    json.dump(dict(video_name="round.mp4", source_clip=S, orig_start_frame=a, orig_end_frame=b,
                   fps=C.FPS, n_frames=n, timeline_segments=tl), open(path, "w"), indent=1)


def render_action(vid, rjson, out):
    tl = json.load(open(rjson))["timeline_segments"]
    cap = cv2.VideoCapture(vid); W = int(cap.get(3)); H = int(cap.get(4)); N = int(cap.get(7))
    ff = C.open_writer(out, W, H); by = H-34; i = 0
    def act(fr):
        for s in tl:
            if s["start_frame"] <= fr <= s["end_frame"]:
                return s["labels"][0]
        return "background"
    while True:
        ok, img = cap.read()
        if not ok:
            break
        a = act(i); c = ACTION_COL.get(a, (200,200,200))
        cv2.rectangle(img, (0,0), (W,46), (0,0,0), -1)
        cv2.putText(img, f"{i/C.FPS:5.1f}s  {a.upper().replace('_',' ')}", (12,34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 3)
        cv2.rectangle(img, (0,by), (W,by+34), (20,20,20), -1)
        for s in tl:
            x0 = int(s["start_frame"]/N*W); x1 = int((s["end_frame"]+1)/N*W)
            cv2.rectangle(img, (x0,by+6), (x1,by+28), ACTION_COL.get(s["labels"][0], (90,90,90)), -1)
        cx = int(i/max(1,N)*W); cv2.line(img, (cx,by), (cx,by+34), (255,255,255), 2)
        C.write_frame(ff, img); i += 1
    cap.release(); C.close_writer(ff); return i


def render_skeleton(vid, frames, out):
    cap = cv2.VideoCapture(vid); W = int(cap.get(3)); H = int(cap.get(4))
    ff = C.open_writer(out, W, H); i = 0
    while True:
        ok, img = cap.read()
        if not ok or i >= len(frames):
            break
        C.draw_interp_skeleton(img, frames[i])
        cv2.rectangle(img, (0,0), (W,40), (0,0,0), -1)
        cv2.putText(img, f"view {frames[i]['view']}  body(cyan)+wilor hands(R green/L red)  yellow=interp(all views)",
                    (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        C.write_frame(ff, img); i += 1
    cap.release(); C.close_writer(ff); return i


def main():
    for d in ("json","vid","vis_round","dwpose","wilor","interpolated_dwpose_wilor","interpolated_dwpose_wilor_vis"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    dw = C.load_dwpose(S); wl = C.load_wilor(S)
    Nfull = len(dw); H, W = dw[0]["frame_dimensions"]
    view_full = C.load_view_array(S, Nfull)
    src = f"{C.VID}/{S}.mp4"
    for i, (a, b, segs) in enumerate(rounds_from_ann(), 1):
        rj = f"{OUT}/json/round{i}.json"; rv = f"{OUT}/vid/round{i}.mp4"
        round_json(segs, a, b, rj)
        C.cut_exact(src, a, b, rv)
        pickle.dump(dw[a:b+1], open(f"{OUT}/dwpose/round{i}_dwpose.pkl", "wb"))
        pickle.dump(wl[a:b+1], open(f"{OUT}/wilor/round{i}_wilor.pkl", "wb"))
        frames = C.build_interpolated(dw[a:b+1], wl[a:b+1], view_full[a:b+1], W, H, BODY_CONF, HAND_CONF)
        pickle.dump(dict(source_clip=S, orig_start_frame=a, orig_end_frame=b, fps=C.FPS,
                         n_frames=b-a+1, body_conf=BODY_CONF, hand_min_conf=HAND_CONF, frames=frames),
                    open(f"{OUT}/interpolated_dwpose_wilor/round{i}.pkl", "wb"))
        na = render_action(rv, rj, f"{OUT}/vis_round/round{i}.mp4")
        ns = render_skeleton(rv, frames, f"{OUT}/interpolated_dwpose_wilor_vis/round{i}.mp4")
        v2 = int((view_full[a:b+1] == 2).sum())
        print(f"round{i}: frames {a}-{b} ({b-a+1}f)  view2={v2}f  vis_action={na}f  vis_skel={ns}f", flush=True)
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
