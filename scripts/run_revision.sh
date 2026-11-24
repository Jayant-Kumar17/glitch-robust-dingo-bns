#!/usr/bin/env bash
# Referee-revision Phase 2. Waits until Stage 3b has exited, then runs R1–R7.
# A failed step does not stop the later steps, except that R4 and R6 are skipped
# when R1 did not produce its result.
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="src:examples"
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
mkdir -p results/revision
exec > >(tee -a results/revision/run.log) 2>&1
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(stamp) waiting for Stage 3b"
while pgrep -f "gated_is_stage3b|run_stage3b" >/dev/null; do
  sleep 300
done
echo "$(stamp) Stage 3b has exited"

echo "$(stamp) START pytest"
if ! python -m pytest -q; then
  echo "$(stamp) STOP pytest failed"
  exit 1
fi
echo "$(stamp) DONE pytest"

step() {
  local name="$1"
  shift
  echo "$(stamp) START $name"
  if "$@"; then
    echo "$(stamp) DONE $name"
    return 0
  else
    echo "$(stamp) FAILED $name"
    return 1
  fi
}

r1_ok=0
if step R1 python -u examples/clean_ref_v2.py --num-samples 100000 --num-processes 8; then
  r1_ok=1
fi
if [ "$r1_ok" = 1 ]; then
  step R6 python -u examples/weights_check.py --num-processes 8 || true
else
  echo "$(stamp) SKIP R6 because R1 failed"
fi
step R2 python -u examples/ungated_is_baseline.py --max-hours 5 --num-processes 8 || true
step R3 python -u examples/gated_is_stage3.py --groups fine --pads 0.1 --outdir results/revision/stage3_fine --max-hours 9 --num-processes 8 || true
if [ "$r1_ok" = 1 ]; then
  step R4 python -u examples/revision_routes.py --max-hours 3 --num-processes 8 || true
else
  echo "$(stamp) SKIP R4 because R1 failed"
fi
step R5 python -u examples/timing.py --num-processes 8 || true
if [ -f examples/synthetic_panel.py ]; then
  step R7 python -u examples/synthetic_panel.py --max-hours 4 --num-processes 8 || true
fi

echo "$(stamp) START analysis"
if python -u examples/revision_analysis.py; then
  echo "$(stamp) DONE analysis"
else
  echo "$(stamp) FAILED analysis"
fi
if compgen -G "results/gated_is_stage3b/cells/*.json" >/dev/null; then
  echo "$(stamp) START fig5_3b"
  if python paper/scripts/make_trust_scatter.py --stage3 results/gated_is_stage3 --stage3b results/gated_is_stage3b --out paper/figures/fig5_trust_scatter_3b.pdf; then
    echo "$(stamp) DONE fig5_3b"
  else
    echo "$(stamp) FAILED fig5_3b"
  fi
else
  echo "$(stamp) SKIP fig5_3b: no Stage 3b cell json"
fi
echo "$(stamp) launcher finished"
