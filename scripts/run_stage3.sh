#!/usr/bin/env bash
# Stage 3 overnight runner: tests -> excerpt diagnosis -> smoke -> full. Unattended; never prompts.
# Usage (repo root, conda env active):
#   export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
#   nohup caffeinate -is bash scripts/run_stage3.sh > results/stage3_runner.log 2>&1 &
set -u
OUT=results/gated_is_stage3
LOG=results/stage3_logs
mkdir -p "$LOG" "$OUT"
NPROC=$(python -c "import os; print(max(2, (os.cpu_count() or 4) - 2))")
stamp () { date '+%Y-%m-%d %H:%M:%S'; }

if [ ! -f results/stress_real_glitches_v3_final/results.csv ]; then
  echo "$(stamp) STOP: results/stress_real_glitches_v3_final/results.csv missing" | tee "$OUT/STOPPED.txt"; exit 1
fi

echo "$(stamp) [1/4] pytest"
if ! python -m pytest -q tests > "$LOG/pytest.log" 2>&1; then
  echo "$(stamp) STOP: tests failed, see $LOG/pytest.log" | tee "$OUT/STOPPED.txt"; exit 1
fi
tail -1 "$LOG/pytest.log"

echo "$(stamp) [2/4] excerpt diagnosis (why v3 has extra quiet excerpts)"
python -u examples/diagnose_excerpts.py > "$LOG/diagnose_excerpts.log" 2>&1 || echo "$(stamp) diagnosis failed (non-fatal)"
head -1 "$LOG/diagnose_excerpts.log"

echo "$(stamp) [3/4] smoke run"
if ! python -u examples/gated_is_stage3.py --smoke --num-processes "$NPROC" --outdir results/_smoke_stage3 \
      > "$LOG/smoke.log" 2>&1; then
  echo "$(stamp) STOP: smoke run crashed, see $LOG/smoke.log" | tee "$OUT/STOPPED.txt"; exit 1
fi
if grep -q "mismatched" results/_smoke_stage3/STAGE3_REPORT.md && ! grep -q ", 0 mismatched" results/_smoke_stage3/STAGE3_REPORT.md; then
  echo "$(stamp) STOP: rebuilt injections do not match Phase C" | tee "$OUT/STOPPED.txt"; exit 1
fi

echo "$(stamp) [4/4] full run with $NPROC processes"
python -u examples/gated_is_stage3.py --num-processes "$NPROC" --max-hours 11 --outdir "$OUT" > "$LOG/full.log" 2>&1
echo "$(stamp) full run exited with code $?"
cat "$OUT/STAGE3_REPORT.md" 2>/dev/null
echo "$(stamp) DONE" > "$OUT/DONE.txt"
