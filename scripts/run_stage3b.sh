#!/usr/bin/env bash
# Stage 3b overnight: rerun Stage 3's untrusted ±0.1 s cells with 1e5 proposal samples.
# Usage (repo root, conda env active):
#   export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
#   nohup caffeinate -is bash scripts/run_stage3b.sh > results/stage3b_runner.log 2>&1 &
set -u
SRC=results/gated_is_stage3
OUT=results/gated_is_stage3b
LOG=results/stage3b_logs
mkdir -p "$LOG" "$OUT"
NPROC=$(python -c "import os; print(max(2, (os.cpu_count() or 4) - 2))")
stamp () { date '+%Y-%m-%d %H:%M:%S'; }

if [ ! -d "$SRC/cells" ]; then
  echo "$(stamp) STOP: $SRC/cells missing" | tee "$OUT/STOPPED.txt"; exit 1
fi
echo "$(stamp) [1/2] pytest"
if ! python -m pytest -q tests > "$LOG/pytest.log" 2>&1; then
  echo "$(stamp) STOP: tests failed, see $LOG/pytest.log" | tee "$OUT/STOPPED.txt"; exit 1
fi
tail -1 "$LOG/pytest.log"

run () {  # run <num_samples>
  python -u examples/gated_is_stage3.py --only-untrusted-from "$SRC" --pads 0.1 \
    --num-samples "$1" --batch-size 5000 --clean-samples 10000 --chunk 20000 \
    --num-processes "$NPROC" --max-hours 11 --outdir "$OUT" >> "$LOG/full.log" 2>&1
}
echo "$(stamp) [2/2] full run, 1e5 samples, $NPROC processes"
run 100000
rc=$?
n_done=$(ls "$OUT"/cells/*.json 2>/dev/null | wc -l | tr -d ' ')
if [ $rc -ne 0 ] && [ "$n_done" = "0" ]; then
  echo "$(stamp) 1e5 run failed before any cell finished (likely memory); retrying with 5e4" | tee -a "$OUT/NOTE.txt"
  run 50000
fi
echo "$(stamp) exited; cells done: $(ls "$OUT"/cells/*.json 2>/dev/null | wc -l | tr -d ' ')"
cat "$OUT/STAGE3_REPORT.md" 2>/dev/null
echo "$(stamp) DONE" > "$OUT/DONE.txt"
