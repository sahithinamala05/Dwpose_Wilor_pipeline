"""STAGE 1: DWPose (body + face + DWPose hands) -> studio dwpose pkls.

UNFILTERED variant: every keypoint DWPose predicts is kept. Nothing is zeroed by a
confidence gate, and the RAW per-keypoint confidences are stored alongside the coords
(`scores`/`face_scores`/`hands_scores`) so any thresholding can be done downstream.

Schema (superset of batch_01 dwpose):
  {'pose': {'bodies': {'candidate', 'subset', 'scores'}, 'hands', 'faces',
            'face_scores', 'hands_scores'},
   'frame_dimensions': [H, W]}
  - 'subset'  : batch_01 index encoding (flat index per body kpt, -1 only if conf <= SCORE_THRESH).
  - 'scores'  : RAW body confidences (N*18,), aligned to 'candidate'.
  - 'face_scores' (N,68), 'hands_scores' (2N,21): raw confidences for faces / hands.

CPU/GPU onnxruntime, thread-capped (else it explodes to ~195 threads and thrashes).
Run one process per video for parallelism. Stage 2 swaps hands -> WiLoR.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys, glob, pickle, argparse, time, traceback
import numpy as np, cv2
cv2.setNumThreads(1)

DWROOT = "/home/ubuntu/us-west-3-fs/sahithi/_entire_code/dwpose_repo/ControlNet-v1-1-nightly"
CK = f"{DWROOT}/annotator/ckpts"
ORT_THREADS = 6
SCORE_THRESH = 0.0   # UNFILTERED: keep every keypoint (was 0.3). Raw scores live in the *_scores fields.
sys.path.insert(0, DWROOT)

_MM = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
_OP = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]

def _sessions():
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = ORT_THREADS; so.inter_op_num_threads = 1
    prov = ['CUDAExecutionProvider', 'CPUExecutionProvider']   # DWPose on GPU (onnxruntime CUDA EP)
    return (ort.InferenceSession(f"{CK}/yolox_l.onnx", sess_options=so, providers=prov),
            ort.InferenceSession(f"{CK}/dw-ll_ucoco_384.onnx", sess_options=so, providers=prov))

def dwpose_frame(img, det_sess, pose_sess):
    from annotator.dwpose.onnxdet import inference_detector
    from annotator.dwpose.onnxpose import inference_pose
    H, W = img.shape[:2]
    det = inference_detector(det_sess, img)
    keypoints, scores = inference_pose(pose_sess, det, img)
    if keypoints is None or len(keypoints) == 0:
        return None
    ki = np.concatenate((keypoints, scores[..., None]), axis=-1)
    neck = np.mean(ki[:, [5, 6]], axis=1)
    neck[:, 2:4] = np.logical_and(ki[:, 5, 2:4] > 0.3, ki[:, 6, 2:4] > 0.3).astype(int)
    ki = np.insert(ki, 17, neck, axis=1)
    ki[:, _OP] = ki[:, _MM]
    candidate, subset = ki[..., :2].copy(), ki[..., 2].copy()   # subset = raw per-keypoint confidence
    nums = candidate.shape[0]
    candidate[..., 0] /= float(W); candidate[..., 1] /= float(H)
    body = candidate[:, :18].reshape(nums * 18, 2)
    # batch_01 index encoding, but WITHOUT the 0.3 drop: every point keeps its flat index
    # (raise SCORE_THRESH to re-enable dropping). Raw confidences are preserved in *_scores below.
    score = subset[:, :18].copy()
    for i in range(len(score)):
        for j in range(len(score[i])):
            score[i][j] = int(18 * i + j) if score[i][j] > SCORE_THRESH else -1
    # candidate coords are intentionally NOT masked -> low-confidence keypoints keep their position
    faces = candidate[:, 24:92]
    hands = np.vstack([candidate[:, 92:113], candidate[:, 113:134]])   # DWPose hands (swapped in stage 2)
    # raw confidences, aligned to each part's keypoint layout (nothing filtered)
    body_scores = subset[:, :18].reshape(nums * 18).copy()
    face_scores = subset[:, 24:92].copy()
    hands_scores = np.vstack([subset[:, 92:113], subset[:, 113:134]])
    return dict(bodies=dict(candidate=body, subset=score, scores=body_scores),
                hands=hands, faces=faces,
                face_scores=face_scores, hands_scores=hands_scores)

def process(vp, out):
    stem = os.path.splitext(os.path.basename(vp))[0]
    det_sess, pose_sess = _sessions()
    cap = cv2.VideoCapture(vp); frames_out = []; t0 = time.time(); i = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        H, W = img.shape[:2]
        pose = dwpose_frame(img, det_sess, pose_sess)
        fr = {'pose': pose, 'frame_dimensions': [H, W]}
        if pose is None: fr['no_dwpose'] = True
        frames_out.append(fr); i += 1
        if i % 200 == 0: print(f"  {stem}: {i} frames ({i/(time.time()-t0):.1f} fps)", flush=True)
    cap.release()
    os.makedirs(out, exist_ok=True)
    pickle.dump(frames_out, open(os.path.join(out, f"{stem}_dwpose.pkl"), 'wb'))
    return f"ok {stem}: {len(frames_out)} frames in {(time.time()-t0)/60:.1f} min"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="/home/ubuntu/us-west-3-fs/sahithi/livedealer_studio_vid_raw/studio_dwpose")
    a = ap.parse_args()
    try: print(process(a.video, a.out), flush=True)
    except Exception: print("ERR\n"+traceback.format_exc(), flush=True)
