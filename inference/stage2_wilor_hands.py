"""STAGE 2: drop DWPose hands, insert ALL WiLoR hands with the COMPLETE raw model output.

Reads the stage-1 dwpose pkls (body/face + DWPose hands) and the source videos, runs WiLoR
(torch/GPU) per frame, and writes pkls where hands carry WiLoR's full output. Every detected
hand is kept (DET_CONF=0.0 in wilor_core); nothing is thresholded away.

Schema per frame:
  {'pose': {'bodies', 'faces',                      # DWPose (raw, unfiltered)
            'hands' (K,21,2 norm image 2D),         # convenience / overlay
            'hands_is_right' [K], 'hands_conf' [K],
            'hands_raw' [K dicts]},                  # COMPLETE WiLoR output per hand:
   'frame_dimensions': [H, W]}
  hands_raw[k] = the model's COMPLETE output dict (native keys), captured generically:
                  pred_cam(3), pred_cam_t(3), focal_length(2), pred_keypoints_3d(21,3),
                  pred_keypoints_2d(21,2), pred_vertices(778,3),
                  pred_mano_params{global_orient,hand_pose,betas}, + any other model key;
                  plus det_box(6 raw YOLO row), bbox(4 px), conf, is_right,
                  pred_cam_t_full(3), keypoints_2d_img(21,2 norm).

Reuses wilor_core. GPU, single process. Run in the `wilor` conda env.
Usage: python stage2_wilor_hands.py [--src DIR] [--dw DIR] [--out DIR] [--no-vertices] [--no-mano]
"""
import os, sys, glob, pickle, argparse, time, traceback
import numpy as np, cv2
cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def process(W, vp, dwp, out):
    stem = os.path.splitext(os.path.basename(vp))[0]
    dw = pickle.load(open(dwp, 'rb')) if os.path.exists(dwp) else None
    cap = cv2.VideoCapture(vp); frames_out = []; i = 0; t0 = time.time()
    while True:
        frames, idxs = [], []
        for _ in range(W.CHUNK):
            ok, fr = cap.read()
            if not ok: break
            frames.append(fr); idxs.append(i); i += 1
        if not frames: break
        hands_per = W.wilor_hands_chunk(frames)
        for img, ii, hands in zip(frames, idxs, hands_per):
            have_dw = dw is not None and ii < len(dw) and dw[ii].get('pose') is not None
            if have_dw:
                pose = dict(dw[ii]['pose'])                    # carry ALL DWpose fields (body/face + raw scores)
                pose.pop('hands_scores', None)                 # DWpose hand scores drop out with DWpose hands
                fdim = dw[ii]['frame_dimensions']
            else:
                pose = {'bodies': None, 'faces': None}; fdim = [img.shape[0], img.shape[1]]
            hn, isr, cf, raw = W.all_wilor_hands(hands)        # DWPose hands replaced by ALL WiLoR hands
            pose['hands'] = hn; pose['hands_is_right'] = isr
            pose['hands_conf'] = cf; pose['hands_raw'] = raw
            fr_out = {'pose': pose, 'frame_dimensions': fdim}
            if not have_dw: fr_out['no_dwpose'] = True
            frames_out.append(fr_out)
    cap.release()
    os.makedirs(out, exist_ok=True)
    pickle.dump(frames_out, open(os.path.join(out, f"{stem}_wilor.pkl"), 'wb'))
    return f"ok {stem}: {len(frames_out)} frames in {(time.time()-t0)/60:.1f} min"

if __name__ == "__main__":
    D = "/home/ubuntu/us-west-3-fs/sahithi/livedealer_studio_vid_raw"
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/ubuntu/us-west-3-fs/sahithi/livedealer_studio_vid/sample")
    ap.add_argument("--dw",  default=f"{D}/studio_dwpose")
    ap.add_argument("--out", default=f"{D}/pkl")
    ap.add_argument("--no-vertices", action="store_true", help="skip the (778,3) MANO mesh per hand")
    ap.add_argument("--no-mano", action="store_true", help="skip the raw MANO params per hand")
    a = ap.parse_args()
    import wilor_core as W                               # imports torch/wilor; CWD must be WILOR
    os.chdir(W.WILOR)                                    # wilor cfg uses ./mano_data, ./pretrained_models
    if a.no_vertices: W.STORE_VERTICES = False
    if a.no_mano:     W.STORE_MANO = False
    W.init()
    vids = sorted(glob.glob(os.path.join(a.src, "*.mkv")) + glob.glob(os.path.join(a.src, "*.mp4")))
    print(f"stage2 WiLoR raw ({len(vids)} videos -> {a.out}) "
          f"vertices={W.STORE_VERTICES} mano={W.STORE_MANO}", flush=True)
    for vp in vids:
        stem = os.path.splitext(os.path.basename(vp))[0]
        dwp = os.path.join(a.dw, f"{stem}_dwpose.pkl")
        try: print(process(W, vp, dwp, a.out), flush=True)
        except Exception: print(f"ERR {stem}\n{traceback.format_exc()}", flush=True)
    print("DONE", flush=True)
