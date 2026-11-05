#!/usr/bin/env bash
# 2-stage DWpose(body/face, raw) + WiLoR(hands, COMPLETE raw output) pipeline for a small set of
# studio recordings. Writes raw pkls to $D/pkl. Run in the `wilor` conda env.  Usage: bash run_all.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D=/home/ubuntu/us-west-3-fs/sahithi/livedealer_studio_vid_raw
SRC=/home/ubuntu/us-west-3-fs/sahithi/livedealer_studio_vid/sample
mkdir -p $D/studio_dwpose $D/pkl $HERE/logs

source ~/miniconda3/etc/profile.d/conda.sh; conda activate wilor
# DWpose runs on GPU via onnxruntime CUDA EP -> needs torch's bundled cuDNN 9 on the lib path
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(ls -d $SP/nvidia/*/lib | paste -sd:):$LD_LIBRARY_PATH"

echo "== STAGE 1: DWpose (body/face + raw scores), one GPU process per video =="
for v in $SRC/*.mkv; do
  s=$(basename "$v" .mkv)
  python "$HERE/stage1_dwpose.py" --video "$v" --out $D/studio_dwpose > "$HERE/logs/dw_$s.log" 2>&1 &
done
wait
echo "== STAGE 2: drop DWpose hands, insert ALL WiLoR hands with full 3D/MANO output =="
python "$HERE/stage2_wilor_hands.py" --src $SRC --dw $D/studio_dwpose --out $D/pkl 2>&1 | tee "$HERE/logs/stage2.log"
echo "DONE -> raw pkls in $D/pkl"
