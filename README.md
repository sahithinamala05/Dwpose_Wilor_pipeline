# dwpose_wilor — baccarat pose & annotation pipeline

Pose extraction and downstream analysis for live-dealer **baccarat** video: per-clip **DWpose** (body
+ face) and **WiLoR** (MANO hands) inference, View-1/View-2 camera segmentation, dealer-identity
segregation, round segmentation, and a per-round action-annotation artifact tree (DWpose body + WiLoR
hands with gap interpolation). Shared logic lives in **`common.py`**.

## Install
```bash
conda create -n wilor python=3.10 && conda activate wilor
pip install -r requirements.txt
sudo apt install ffmpeg                       # system dependency (all video I/O)
# onnxruntime-gpu needs cuDNN — use the copy bundled with torch:
export LD_LIBRARY_PATH=$(python -c "import torch,os;print(os.path.dirname(torch.__file__)+'/lib')"):$LD_LIBRARY_PATH
```

## Configure
All paths derive from one dataset root, set in `common.py` and overridable per-run:
```bash
export BACCARAT_ROOT=/path/to/baccarat        # default: /home/ubuntu/us-west-3-fs/sahithi/baccarat
```
Expected layout under `$BACCARAT_ROOT` is documented in `$BACCARAT_ROOT/STRUCTURE.md`.

## Quickstart
```bash
# 1. pose inference over the source clips
bash inference/run_batch.sh --mode both --input $BACCARAT_ROOT/vid   # -> dwpose/, wilor/
# 2. camera-view segmentation
python pose/face_view_segment.py                      # -> view_annot/
# 3. segregate footage by dealer identity (InsightFace GPU)
python dealers/single_dealer_filter.py --limit 300    # -> dealers/vid_single_dealer/dealer_<k>/
# 4. segment a dealer's clips into rounds
python rounds/segment_rounds.py --dealer dealer_4     # -> annotation/annotate_batch/
# 5. after manual action labeling (see ../baccarat_annotator), build the round artifact tree
python rounds/build_round_artifacts.py                # -> annotation/action_annot_round/dealer4/
python rounds/build_view_labels.py                    # add View-1/2 labeling
```

## common.py (shared library)
Imported as `import common as C`. Holds everything that used to be copy-pasted:
- **paths** — `C.B, VID, DWPOSE, WILOR, VIEW_ANNOT, VIS, ANNOT, DEALERS, FPS`
- **skeleton** — `BODY_LIMBS, HAND_LIMBS, R_ELBOW/L_ELBOW, R_WRIST/L_WRIST`, colors
- **pose IO** — `load_dwpose/load_wilor`, `body_points`, `best_hands`, `all_hands`
- **interpolation** — `interp_runs` (per-view-run gap fill), `build_interpolated` (DWpose body + WiLoR hands, elbows+hands interpolated)
- **views** — `load_view_array`, `view_segments`
- **video** — `open_writer/write_frame/close_writer`, `cut_exact` (frame-exact), `cut_segment` (time-based), `draw_interp_skeleton`
- **cards** — `card_signal` (table card-presence)

## Layout
`common.py` sits at the repo root; the scripts are grouped by stage. Scripts in subfolders import the
shared library via a one-line path bootstrap at the top.

### `inference/` — model workers (standalone, no `common` dependency)
- `stage1_dwpose.py`, `wilor_core.py`, `stage2_wilor_hands.py` — the DWpose/WiLoR workers
- `batch_pipeline.py` — orchestrates inference over `vid/` → `dwpose/`, `wilor/` (`--mode dwpose|wilor|both`)
- `run_batch.sh`, `run_all.sh` — launchers (self-relative; set conda + cuDNN lib path)

### `pose/` — pose-derived products
- `face_view_segment.py` — View-1/2 segmentation → `view_annot/`
- `stitch_view1.py` — switch-free View-1 stream → `view1_stitched/`
- `forearm_interp.py` — forearm (elbow+wrist) with view-2 interpolation → `forearm/` *(predates common)*
- `vis_scripts/` — `viz_dwpose`, `viz_wilor`, `viz_dwpose_byview` overlays → `vis/`

### `dealers/` — dealer identity (InsightFace-GPU; standalone)
- `single_dealer_filter.py` — segregate footage by dealer → `dealers/vid_single_dealer/dealer_<k>/`
- `dealer_change.py`, `dealer_change_temporal.py` — (earlier) dealer-change detection

### `rounds/` — round segmentation & per-round action artifacts
- `segment_rounds.py` — card-signal round segmentation → `annotation/annotate_batch/`
- `annotate_view1.py` — view-1-stream action labels → `annotation/action_view1/`
- `build_round_artifacts.py` — the dealer4 per-round tree (json/vid/vis/dwpose/wilor/interpolated/+vis)
- `redo_interp.py` — regenerate just the interpolated pose + its viz
- `build_view_labels.py` — add View-1/2 labeling to the round tree

The manual action annotator (Flask) lives in `../baccarat_annotator/`.
