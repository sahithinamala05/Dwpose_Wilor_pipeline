"""WiLoR hand machinery (consolidated, COMPLETE RAW OUTPUT).

The reusable WiLoR path shared by stage2_wilor_hands.py and batch_pipeline.py. Exposes:
  init()                      -- load WiLoR + YOLO detector once (per process)
  wilor_hands_chunk(frames)   -- per frame: EVERY detected hand -> full raw model-output dict
  all_wilor_hands(hands)      -- repackage one frame's hands -> (kpts2d_img, is_right, conf, raw)

UNFILTERED: DET_CONF = 0.0, so the detector keeps every hand box it produces -- no confidence
gate drops low-score detections. For every hand we store the model's COMPLETE output dict,
captured GENERICALLY (every key WiLoR emits, by its native name), plus the detector's full raw
box and a couple of derived conveniences:
  <all model keys>: pred_cam (3), pred_cam_t (3), focal_length (2),
                    pred_keypoints_3d (21,3), pred_keypoints_2d (21,2), pred_vertices (778,3),
                    pred_mano_params {global_orient (1,3,3), hand_pose (15,3,3), betas (10)},
                    ... and any other key the model returns (nothing hand-picked).
  det_box (6,) raw YOLO row [x1,y1,x2,y2,conf,cls], bbox (4,) px, conf (float), is_right (bool),
  pred_cam_t_full (3,) full-image cam translation, keypoints_2d_img (21,2 norm, projected to image).

pred_vertices + pred_mano_params are large; set STORE_VERTICES / STORE_MANO = False (stage2
--no-vertices / --no-mano) to drop them for lighter pkls. Default keeps everything.

Needs the `wilor` package + weights from the Wilor repo (WILOR). Run with CWD = WILOR (its cfg
uses ./mano_data, ./pretrained_models), in the `wilor` conda env. Imports torch/wilor at import
time, so only the WiLoR worker should import it -- keep it out of the DWPose (onnxruntime)
process (that combo deadlocks).
"""
import os, sys
WILOR = "/home/ubuntu/us-west-3-fs/sahithi/_entire_code/Wilor"
sys.path.insert(0, WILOR)
import numpy as np, cv2, torch
from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel
from torch.utils.data import default_collate
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full

torch.serialization.add_safe_globals([PoseModel])
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})

CKPT = f"{WILOR}/pretrained_models/wilor_final.ckpt"
CFG  = f"{WILOR}/pretrained_models/model_config.yaml"
DETECTOR = f"{WILOR}/pretrained_models/detector.pt"
CHUNK = 64
WILOR_BS = 128
DET_CONF = 0.0          # UNFILTERED: keep every detected hand box (was 0.3). Confidence is stored, not used to drop.
STORE_VERTICES = True   # store raw MANO mesh (778,3) per hand -- large
STORE_MANO = True       # store raw MANO params (global_orient, hand_pose, betas) per hand

_model = _cfg = _det = _dev = None
def init():
    global _model, _cfg, _det, _dev
    if _model is not None:
        return
    _dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model, _cfg = load_wilor(checkpoint_path=CKPT, cfg_path=CFG)
    _det = YOLO(DETECTOR)
    _model = _model.to(_dev).eval()

def _project(points, cam_t, f, img_res):
    cc = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3); K[0, 0] = f; K[1, 1] = f; K[0, 2] = cc[0]; K[1, 2] = cc[1]
    p = points + cam_t
    p = p / p[..., -1:]
    return (K @ p.T).T[..., :-1]

def _to_np(v):
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    if isinstance(v, dict):
        return {k: _to_np(x) for k, x in v.items()}
    return v

# large model-output keys to drop when STORE_VERTICES / STORE_MANO are off
_HEAVY = {'pred_vertices': 'STORE_VERTICES', 'pred_mano_params': 'STORE_MANO'}

def wilor_hands_chunk(frames):
    """Per frame -> list of per-hand dicts holding the model's COMPLETE output (see module docstring)."""
    out = [[] for _ in frames]
    items, owner, det_rows, dims = [], [], [], []
    for fi, f in enumerate(frames):
        H, W = f.shape[:2]
        dets = _det(f, conf=DET_CONF, verbose=False)[0]        # DET_CONF=0.0 -> keep every detection
        bboxes, is_right, rows = [], [], []
        for d in dets:
            b = d.boxes.data.cpu().detach().squeeze().numpy()  # full raw row [x1,y1,x2,y2,conf,cls]
            is_right.append(d.boxes.cls.cpu().detach().squeeze().item())
            rows.append(np.asarray(b, dtype=np.float32))
            bboxes.append(b[:4].tolist())
        if not bboxes:
            continue
        ds = ViTDetDataset(_cfg, f, np.stack(bboxes), np.stack(is_right), rescale_factor=1.0)
        for i in range(len(ds)):
            items.append(ds[i]); owner.append(fi); det_rows.append(rows[i]); dims.append((W, H))
    if not items:
        return out
    for s in range(0, len(items), WILOR_BS):
        batch = recursive_to(default_collate(items[s:s + WILOR_BS]), _dev)
        with torch.no_grad():
            res = _model(batch)
        B = batch['img'].shape[0]
        # snapshot the COMPLETE raw model output (every key, before any post-processing)
        res_np = {}
        for k, v in res.items():
            if k in _HEAVY and not globals()[_HEAVY[k]]:
                continue                                        # skip vertices / mano when disabled
            res_np[k] = _to_np(v)
        # standard WiLoR post-processing -> full-image camera translation (to project 3D -> image px)
        pc = res['pred_cam'].clone(); pc[:, 1] = (2 * batch['right'] - 1) * pc[:, 1]
        bc = batch['box_center'].float(); bs = batch['box_size'].float(); isz = batch['img_size'].float()
        f = _cfg.EXTRA.FOCAL_LENGTH / _cfg.MODEL.IMAGE_SIZE * isz.max()
        ct = cam_crop_to_full(pc, bc, bs, isz, f).detach().cpu().numpy()
        k3d = res['pred_keypoints_3d'].detach().cpu().numpy()
        rights = batch['right'].cpu().numpy()
        for n in range(B):
            idx = s + n; W, H = dims[idx]; rh = rights[n]
            hand = {}
            for k, v in res_np.items():                         # every model output, sliced to this hand
                if isinstance(v, dict):
                    hand[k] = {kk: (x[n] if isinstance(x, np.ndarray) and x.ndim and x.shape[0] == B else x)
                               for kk, x in v.items()}
                elif isinstance(v, np.ndarray) and v.ndim and v.shape[0] == B:
                    hand[k] = v[n]
                else:
                    hand[k] = v
            j = k3d[n].copy(); j[:, 0] = (2 * rh - 1) * j[:, 0]          # flip L->common frame for projection
            k2d_px = np.asarray(_project(j, ct[n], f, isz[n])[:, :2], dtype=np.float64)
            hand['is_right'] = bool(rh)
            hand['conf'] = float(det_rows[idx][4]) if det_rows[idx].shape[0] > 4 else float('nan')
            hand['bbox'] = det_rows[idx][:4]
            hand['det_box'] = det_rows[idx]                     # full raw detector row [x1,y1,x2,y2,conf,cls]
            hand['pred_cam_t_full'] = ct[n]                     # full-image translation (pairs with keypoints_2d_img)
            hand['keypoints_2d_img'] = k2d_px / [W, H]          # normalized image coords (for overlay)
            out[owner[idx]].append(hand)
    return out

def all_wilor_hands(hands):
    """Repackage one frame's hands -> (kpts2d_img (K,21,2 norm), is_right [K], conf [K], raw [K dicts])."""
    if not hands:
        return np.zeros((0, 21, 2), dtype=np.float64), [], [], []
    kpts = np.stack([np.asarray(h['keypoints_2d_img'], dtype=np.float64) for h in hands])
    return kpts, [bool(h['is_right']) for h in hands], [float(h['conf']) for h in hands], hands
