"""OPTIMIZED 2-STAGE BATCH RUNNER -- DWpose(body/face, raw) + WiLoR(hands, COMPLETE raw output)
for THOUSANDS of videos.

Produces `pkl/<stem>_dwpose.pkl` (DWPose body/face + every WiLoR hand with full 3D/MANO output)
for every video under --input, in one streaming multi-worker pipeline. Keeps the stage-1
intermediate (studio_dwpose/). No fill stage, no viz -- raw model output only.

WHY THIS SHAPE:
  * Each stage is a POOL of long-lived workers that load their model ONCE and stream many videos
    through a queue -- not one process per video (which would reload DWpose/WiLoR thousands of times).
  * The two stages OVERLAP: stage-2 starts on a clip the moment stage-1 finishes it. The GPU stays
    busy instead of idling between phases.
  * DWpose (onnxruntime-CUDA) and WiLoR (torch-CUDA) live in SEPARATE processes -- each worker lazily
    imports only its own stage's framework, so they never co-load (that combo deadlocks).
  * RESUMABLE + ATOMIC: a video is enqueued at the earliest stage whose output is missing; every pkl
    is written to a tmp file then os.replace()'d, so a crash never leaves a half-written "done" file.

Run in the `wilor` conda env via run_batch.sh (sets the cuDNN lib path DWpose-on-GPU needs).
  bash run_batch.sh --input /path/to/videos [--dw-workers 4 --wilor-workers 3]
"""
import os, sys, glob, pickle, time, argparse, traceback, threading
import multiprocessing as mp

HERE   = os.path.dirname(os.path.abspath(__file__))
WILOR  = "/home/ubuntu/us-west-3-fs/sahithi/_entire_code/Wilor"
D_ROOT = "/home/ubuntu/us-west-3-fs/sahithi/livedealer_studio_vid_raw"
S1_SUFFIX = "_dwpose.pkl"   # DWpose stage output
S2_SUFFIX = "_wilor.pkl"    # WiLoR stage output
VIDEXTS = (".mkv", ".mp4")


# ------------------------------------------------------------------ helpers
def _cap_threads(ort_threads):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(ort_threads)


def _atomic_dump(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)          # atomic on the same filesystem


def _load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _inc(counter, n=1):
    with counter.get_lock():
        counter.value += n


# ------------------------------------------------------------------ STAGE 1: DWpose (onnxruntime GPU)
def dw_worker(wid, gpu, q1, q2, cfg, ok_c, err_c, fails):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _cap_threads(cfg["ort_threads"])
    import numpy as np, cv2                    # noqa
    cv2.setNumThreads(1)
    dw = _load_module(os.path.join(HERE, "stage1_dwpose.py"), "dw_stage1")
    dw.ORT_THREADS = cfg["ort_threads"]
    det_sess, pose_sess = dw._sessions()       # DWpose loaded ONCE for this worker
    s1_dir = cfg["s1_dir"]
    print(f"[dw{wid}] ready on gpu{gpu}", flush=True)
    while True:
        item = q1.get()
        if item is None:
            break
        stem, video = item["stem"], item["video"]
        t0 = time.time()
        try:
            frames_out = []
            cap = cv2.VideoCapture(video)
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                H, W = img.shape[:2]
                pose = dw.dwpose_frame(img, det_sess, pose_sess)
                fr = {"pose": pose, "frame_dimensions": [H, W]}
                if pose is None:
                    fr["no_dwpose"] = True
                frames_out.append(fr)
            cap.release()
            _atomic_dump(frames_out, os.path.join(s1_dir, stem + S1_SUFFIX))
            _inc(ok_c)
            if cfg.get("handoff", True):
                q2.put(item)                   # hand off to WiLoR stage (only in pipelined 'both' mode)
            if ok_c.value % 50 == 0:
                print(f"[dw{wid}] {stem}: {len(frames_out)}f {(time.time()-t0):.1f}s", flush=True)
        except Exception:
            _inc(err_c); fails.append(f"stage1\t{stem}\t{traceback.format_exc().splitlines()[-1]}")
            print(f"[dw{wid}] FAIL {stem}\n{traceback.format_exc()}", flush=True)


# ------------------------------------------------------------------ STAGE 2: WiLoR hands, raw (torch GPU)
def wilor_worker(wid, gpu, q2, cfg, ok_c, err_c, fails):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _cap_threads(cfg["ort_threads"])
    import numpy as np, cv2                    # noqa
    cv2.setNumThreads(1)
    os.chdir(WILOR)                            # cfg uses ./mano_data, ./pretrained_models
    sys.path.insert(0, HERE)
    import wilor_core as Wm                    # consolidated WiLoR path (torch); loaded here only
    Wm.STORE_VERTICES = cfg["store_vertices"]; Wm.STORE_MANO = cfg["store_mano"]
    Wm.DET_CONF = cfg["det_conf"]              # detector confidence gate for this run
    Wm.init()                                  # WiLoR + detector loaded ONCE for this worker
    s1_dir, s2_dir = cfg["s1_dir"], cfg["s2_dir"]
    print(f"[wilor{wid}] ready on gpu{gpu} (vertices={Wm.STORE_VERTICES} mano={Wm.STORE_MANO})", flush=True)
    while True:
        item = q2.get()
        if item is None:
            break
        stem, video = item["stem"], item["video"]
        t0 = time.time()
        try:
            merge = cfg["merge_dwpose"]                        # False -> INDEPENDENT hands-only output
            dwp = os.path.join(s1_dir, stem + S1_SUFFIX)
            dw = pickle.load(open(dwp, "rb")) if (merge and os.path.exists(dwp)) else None
            cap = cv2.VideoCapture(video); frames_out = []; i = 0
            while True:
                frames, idxs = [], []
                for _ in range(Wm.CHUNK):
                    ok, fr = cap.read()
                    if not ok:
                        break
                    frames.append(fr); idxs.append(i); i += 1
                if not frames:
                    break
                hands_per = Wm.wilor_hands_chunk(frames)
                for img, ii, hands in zip(frames, idxs, hands_per):
                    hn, isr, cf, raw = Wm.all_wilor_hands(hands)
                    if merge:                                  # carry DWpose body/face in (pipelined 'both')
                        have_dw = dw is not None and ii < len(dw) and dw[ii].get("pose") is not None
                        if have_dw:
                            pose = dict(dw[ii]["pose"]); pose.pop("hands_scores", None)
                            fdim = dw[ii]["frame_dimensions"]
                        else:
                            pose = {"bodies": None, "faces": None}; fdim = [img.shape[0], img.shape[1]]
                        pose["hands"] = hn; pose["hands_is_right"] = isr
                        pose["hands_conf"] = cf; pose["hands_raw"] = raw
                        fr_out = {"pose": pose, "frame_dimensions": fdim}
                        if not have_dw:
                            fr_out["no_dwpose"] = True
                    else:                                      # INDEPENDENT: WiLoR hands only, no DWpose merge
                        fr_out = {"pose": {"hands": hn, "hands_is_right": isr,
                                           "hands_conf": cf, "hands_raw": raw},
                                  "frame_dimensions": [img.shape[0], img.shape[1]]}
                    frames_out.append(fr_out)
            cap.release()
            _atomic_dump(frames_out, os.path.join(s2_dir, stem + S2_SUFFIX))
            _inc(ok_c)
            if ok_c.value % 20 == 0:
                print(f"[wilor{wid}] {stem}: {len(frames_out)}f {(time.time()-t0):.1f}s", flush=True)
        except Exception:
            _inc(err_c); fails.append(f"stage2\t{stem}\t{traceback.format_exc().splitlines()[-1]}")
            print(f"[wilor{wid}] FAIL {stem}\n{traceback.format_exc()}", flush=True)


# ------------------------------------------------------------------ orchestration
def enumerate_videos(input_dir, exts, limit):
    seen, vids, dups = {}, [], 0
    for root, _, files in os.walk(input_dir):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in exts:
                continue
            stem = os.path.splitext(fn)[0]
            path = os.path.join(root, fn)
            if stem in seen:                     # stem is the pkl key -> must be unique
                dups += 1
                print(f"WARN duplicate stem '{stem}' -> skipping {path} (keeping {seen[stem]})", flush=True)
                continue
            seen[stem] = path
            vids.append({"stem": stem, "video": path})
    vids.sort(key=lambda x: x["stem"])
    if limit:
        vids = vids[:limit]
    return vids, dups


def route(vids, cfg, redo, mode):
    """Assign each video to a stage queue, skipping ones already done (resume)."""
    q1i, q2i, done = [], [], 0
    for it in vids:
        s = it["stem"]
        f_s1 = os.path.join(cfg["s1_dir"], s + S1_SUFFIX)
        f_s2 = os.path.join(cfg["s2_dir"], s + S2_SUFFIX)
        if mode == "dwpose":                              # independent DWpose job
            if not redo and os.path.exists(f_s1): done += 1
            else: q1i.append(it)
        elif mode == "wilor":                             # independent WiLoR job
            if not redo and os.path.exists(f_s2): done += 1
            else: q2i.append(it)
        else:                                             # 'both' = pipelined (stage2 resumes from stage1)
            if not redo and os.path.exists(f_s2):
                done += 1
            elif not redo and os.path.exists(f_s1):
                q2i.append(it)
            else:
                q1i.append(it)
    return q1i, q2i, done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="/home/ubuntu/us-west-3-fs/sahithi/livedealer_studio_vid/sample",
                    help="dir to scan (recursively) for videos")
    ap.add_argument("--mode", choices=("both", "dwpose", "wilor"), default="both",
                    help="both=pipelined (WiLoR merges DWpose body/face); dwpose/wilor=independent single-model job "
                         "(WiLoR output is hands-only)")
    ap.add_argument("--out-root", default=D_ROOT, help="root holding studio_dwpose/ and pkl/")
    ap.add_argument("--s1-dir", default=None, help="stage-1 DWpose output dir (overrides out-root/studio_dwpose)")
    ap.add_argument("--s2-dir", default=None, help="stage-2 WiLoR output dir (overrides out-root/pkl)")
    ap.add_argument("--det-conf", type=float, default=0.0, help="YOLO hand-detector confidence gate (0.0 = keep all)")
    ap.add_argument("--exts", default=",".join(VIDEXTS), help="comma list of video extensions")
    ap.add_argument("--gpus", default="0", help="comma list of GPU ids to spread workers across")
    ap.add_argument("--dw-workers", type=int, default=4, help="DWpose (stage1) processes")
    ap.add_argument("--wilor-workers", type=int, default=3, help="WiLoR (stage2) processes")
    ap.add_argument("--ort-threads", type=int, default=4, help="intra-op / BLAS threads per worker")
    ap.add_argument("--no-vertices", action="store_true", help="skip the (778,3) MANO mesh per hand")
    ap.add_argument("--no-mano", action="store_true", help="skip the raw MANO params per hand")
    ap.add_argument("--limit", type=int, default=0, help="cap number of videos (0 = all)")
    ap.add_argument("--redo", action="store_true", help="reprocess even if outputs exist")
    ap.add_argument("--dry-run", action="store_true", help="scan + route only, spawn nothing")
    a = ap.parse_args()

    exts = tuple(e if e.startswith(".") else "." + e for e in a.exts.lower().split(","))
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip() != ""]
    cfg = dict(s1_dir=a.s1_dir or os.path.join(a.out_root, "studio_dwpose"),
               s2_dir=a.s2_dir or os.path.join(a.out_root, "pkl"),
               ort_threads=a.ort_threads, det_conf=a.det_conf,
               store_vertices=not a.no_vertices, store_mano=not a.no_mano,
               handoff=(a.mode == "both"),               # dw hands off to wilor only when pipelined
               merge_dwpose=(a.mode == "both"))          # wilor merges body/face only when pipelined
    for d in (cfg["s1_dir"], cfg["s2_dir"]):
        os.makedirs(d, exist_ok=True)

    vids, dups = enumerate_videos(a.input, exts, a.limit)
    q1i, q2i, done = route(vids, cfg, a.redo, a.mode)
    todo = len(q1i) + len(q2i)
    print(f"mode={a.mode} videos={len(vids)} dup_stems={dups} already_done={done} todo={todo} "
          f"(q1={len(q1i)} q2={len(q2i)})", flush=True)
    print(f"workers: dw={a.dw_workers} wilor={a.wilor_workers} on gpus={gpus} "
          f"det_conf={cfg['det_conf']} vertices={cfg['store_vertices']} mano={cfg['store_mano']}", flush=True)
    out_desc = cfg["s1_dir"] if a.mode == "dwpose" else cfg["s2_dir"] if a.mode == "wilor" else f"s1={cfg['s1_dir']} s2={cfg['s2_dir']}"
    print(f"out: {out_desc}", flush=True)
    if a.dry_run or todo == 0:
        print("nothing to do" if todo == 0 else "dry-run: exiting", flush=True)
        return

    ctx = mp.get_context("spawn")              # spawn: required for CUDA in child processes
    q1, q2 = ctx.Queue(), ctx.Queue()
    mgr = ctx.Manager(); fails = mgr.list()
    C = {k: ctx.Value("i", 0) for k in ("s1ok", "s1e", "s2ok", "s2e")}

    def spawn(fn, n, extra):
        ps = []
        for i in range(n):
            gpu = gpus[i % len(gpus)]
            p = ctx.Process(target=fn, args=(i, gpu, *extra), daemon=False)
            p.start(); ps.append(p)
        return ps
    dw_ps    = spawn(dw_worker,    a.dw_workers,    (q1, q2, cfg, C["s1ok"], C["s1e"], fails)) if a.mode in ("both", "dwpose") else []
    wilor_ps = spawn(wilor_worker, a.wilor_workers, (q2, cfg, C["s2ok"], C["s2e"], fails)) if a.mode in ("both", "wilor") else []

    # enqueue initial work at each stage entry point
    for it in q1i: q1.put(it)
    for it in q2i: q2.put(it)

    # heartbeat: print running totals until stopped
    fin_key, err_key = ("s1ok", "s1e") if a.mode == "dwpose" else ("s2ok", "s2e")
    stop = threading.Event(); t_start = time.time()
    def monitor():
        while not stop.wait(20):
            el = (time.time() - t_start) / 60
            fin = C[fin_key].value
            rate = fin / el if el > 0 else 0
            print(f"  [+{el:.1f}m] s1={C['s1ok'].value}/{C['s1e'].value}e s2={C['s2ok'].value}/{C['s2e'].value}e "
                  f"final={fin}/{todo} ({C[err_key].value}e) ~{rate:.1f} vid/min", flush=True)
    mon = threading.Thread(target=monitor, daemon=True); mon.start()

    # sentinel chain: sentinel a queue only once ALL its producers have finished
    if dw_ps:
        for _ in dw_ps: q1.put(None)
        for p in dw_ps: p.join()                 # stage-1 producers done -> (in 'both') q2 fully populated
    if wilor_ps:
        for _ in wilor_ps: q2.put(None)
        for p in wilor_ps: p.join()
    stop.set()

    if len(fails):
        fp = os.path.join(a.out_root, "batch_failures.txt")
        with open(fp, "w") as f:
            f.write("\n".join(fails) + "\n")
        print(f"failures logged -> {fp}", flush=True)

    el = (time.time() - t_start) / 60
    print(f"DONE in {el:.1f}m | final={C[fin_key].value}/{todo} | "
          f"errs s1={C['s1e'].value} s2={C['s2e'].value}", flush=True)


if __name__ == "__main__":
    main()
