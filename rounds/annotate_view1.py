"""Write Claude's visual action labels for a stitched View-1 stream + map them back to original frames + render.

Labels were assigned by visually inspecting ~1 fps montages of the View-1-only (switch-free) stitched
video. Classes: wait_for_bets, clean_hand, hit (dealing = hit), discard, background (dealer change / settling).

Reads <stem>_view1_map.json (stitched_to_orig). Writes:
  action_view1/<stem>_actions.json     segments in stitched-time AND original-frame ranges
  vis/action_view1/<stem>_actions.mp4  label bar + current-action overlay on the stitched video
"""
import os, json, argparse, subprocess
import numpy as np, cv2; cv2.setNumThreads(1)

B = "/home/ubuntu/us-west-3-fs/sahithi/baccarat"
MAPD, STITCHED, OUTJ, OUTV = f"{B}/view1_stitched", f"{B}/view1_stitched", f"{B}/annotation/action_view1", f"{B}/vis/action_view1"
COL = {"wait_for_bets": (255,170,0), "clean_hand": (180,180,180), "hit": (0,220,0),
       "discard": (0,140,255), "background": (90,90,90)}

# (start_s, end_s, action) in STITCHED time for 2025-12-15_19-20-48.
# Method: card-presence signal (central table ROI white-blob area) locates the 5 rounds; dense 3fps
# montages at each round-end pin the DISCARD sweep (dealer gathers cards to center -> table clears).
# hit = deal+reveal (cards being dealt/on table), discard = the clearing sweep, wait = idle empty table.
LABELS = {
 "2025-12-15_19-20-48": [
    (0.0,   6.0,  "wait_for_bets"),
    (6.0,  13.0,  "hit"),            # round 1 deal+reveal (brunette)
    (13.0, 14.6,  "discard"),        # R1 sweep -> clear
    (14.6, 25.0,  "wait_for_bets"),
    (25.0, 30.5,  "hit"),            # round 2
    (30.5, 32.2,  "discard"),        # R2 sweep
    (32.2, 45.0,  "wait_for_bets"),
    (45.0, 48.6,  "hit"),            # round 3
    (48.6, 49.9,  "discard"),        # R3 sweep
    (49.9, 60.0,  "wait_for_bets"),
    (60.0, 67.9,  "hit"),            # round 4
    (67.9, 69.3,  "discard"),        # R4 sweep
    (69.3, 73.0,  "wait_for_bets"),
    (73.0, 90.0,  "background"),     # DEALER CHANGE brunette->blonde + settling in
    (90.0, 94.0,  "wait_for_bets"),
    (94.0, 103.6, "hit"),            # round 5 deal (blonde)
    (103.6,105.3, "discard"),        # R5 sweep
    (105.3,119.0, "wait_for_bets"),
    (119.0,999.0, "hit"),            # next deal begins
 ],
}


def build(stem):
    m = json.load(open(f"{MAPD}/{stem}_view1_map.json"))
    fps = m["fps"]; s2o = m["stitched_to_orig"]; Ns = len(s2o)
    labs = LABELS[stem]
    segs = []
    for s, e, a in labs:
        sf = int(round(s*fps)); ef = min(Ns-1, int(round(e*fps))-1)
        if sf > ef or sf >= Ns:
            continue
        segs.append(dict(action=a, stitched_start_s=round(sf/fps,2), stitched_end_s=round(ef/fps,2),
                         stitched_start_frame=sf, stitched_end_frame=ef,
                         orig_start_frame=s2o[sf], orig_end_frame=s2o[ef]))
    rep = dict(clip=stem, stream="view1_stitched", fps=round(fps,3), n_frames=Ns,
               duration_s=round(Ns/fps,2), classes=list(COL.keys()),
               method="claude_visual + card_presence_signal (round boundaries) + 3fps round-end review (discard)",
               note="labels on switch-free View-1 stream; orig_* map back to source video frames",
               segments=segs)
    return rep, fps, Ns


def action_at(fr, segs):
    for s in segs:
        if s["stitched_start_frame"] <= fr <= s["stitched_end_frame"]:
            return s["action"]
    return segs[-1]["action"]


def render(stem, rep, fps, out_mp4):
    cap = cv2.VideoCapture(f"{STITCHED}/{stem}_view1.mp4"); W=int(cap.get(3)); H=int(cap.get(4)); N=int(cap.get(7))
    segs = rep["segments"]; dur = rep["duration_s"]
    ff = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24",
        "-s",f"{W}x{H}","-r",str(int(round(fps))),"-i","pipe:0","-c:v","libx264","-pix_fmt","yuv420p",
        "-crf","22","-movflags","+faststart",out_mp4], stdin=subprocess.PIPE)
    BARY=H-40; i=0
    while True:
        ok,img=cap.read()
        if not ok: break
        a=action_at(i,segs); c=COL[a]
        cv2.rectangle(img,(0,0),(W,48),(0,0,0),-1)
        cv2.putText(img,f"{i/fps:5.1f}s   {a.upper().replace('_',' ')}",(12,36),cv2.FONT_HERSHEY_SIMPLEX,1.0,c,3)
        cv2.rectangle(img,(0,BARY),(W,BARY+40),(20,20,20),-1)
        for s in segs:
            x0=int(s["stitched_start_s"]/dur*W); x1=int(s["stitched_end_s"]/dur*W)
            cv2.rectangle(img,(x0,BARY+8),(x1,BARY+32),COL[s["action"]],-1)
        cx=int(i/max(1,N)*W); cv2.line(img,(cx,BARY),(cx,BARY+40),(255,255,255),2)
        ff.stdin.write(np.ascontiguousarray(img).tobytes()); i+=1
    cap.release(); ff.stdin.close(); ff.wait()
    return i


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stem", default="2025-12-15_19-20-48")
    ap.add_argument("--no-viz", action="store_true"); a = ap.parse_args()
    os.makedirs(OUTJ, exist_ok=True); os.makedirs(OUTV, exist_ok=True)
    rep, fps, Ns = build(a.stem)
    json.dump(rep, open(f"{OUTJ}/{a.stem}_actions.json","w"), indent=2)
    from collections import Counter
    cc = Counter();
    for s in rep["segments"]: cc[s["action"]] += (s["stitched_end_frame"]-s["stitched_start_frame"]+1)
    print(f"{a.stem}: {len(rep['segments'])} segs over {rep['duration_s']}s")
    for k,v in cc.items(): print(f"   {k:14s} {v/fps:6.1f}s")
    if not a.no_viz:
        w = render(a.stem, rep, fps, f"{OUTV}/{a.stem}_actions.mp4")
        print(f"   viz -> {OUTV}/{a.stem}_actions.mp4 ({w}f)")
    print(f"   json -> {OUTJ}/{a.stem}_actions.json")
