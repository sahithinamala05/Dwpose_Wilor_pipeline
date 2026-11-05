#!/usr/bin/env bash
# Optimized 2-stage batch pipeline for THOUSANDS of videos: DWpose(body/face, raw) +
# WiLoR(hands, COMPLETE raw output) -> pkl/. Keeps the stage-1 intermediate (studio_dwpose/).
# Run in the `wilor` env.
#
# Usage:
#   bash run_batch.sh --input /path/to/videos              # scan recursively, process all
#   bash run_batch.sh --input DIR --wilor-workers 4        # push more throughput on the H100
#   bash run_batch.sh --input DIR --no-vertices --no-mano  # lighter pkls (drop mesh + MANO params)
#   bash run_batch.sh --input DIR --dry-run                # just scan + show the plan
#   nohup bash run_batch.sh --input DIR > batch.out 2>&1 & # detach for a long run
#
# Resumable: re-running skips videos whose pkl/ output already exists.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh; conda activate wilor
# DWpose runs on GPU via onnxruntime CUDA EP -> needs torch's bundled cuDNN 9 on the lib path
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(ls -d $SP/nvidia/*/lib | paste -sd:):$LD_LIBRARY_PATH"

exec python "$HERE/batch_pipeline.py" "$@"
