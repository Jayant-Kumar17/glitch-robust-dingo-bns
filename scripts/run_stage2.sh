#!/usr/bin/env bash
# Stage 2 overnight runner: tests -> smoke run -> full run. Unattended; never prompts.
# Usage (repo root, conda env active):
#   export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
#   nohup caffeinate -is bash scripts/run_stage2.sh /abs/path/L-L1_GWOSC_4KHZ_R1-1187006835-4096.hdf5 > results/stage2_runner.log 2>&1 &
set -u
L1_HDF5="${1:-}"
OUT=results/gated_is_stage2
LOG=results/stage2_logs
mkdir -p "$LOG" "$OUT"
NPROC=$(python -c "import os; print(max(2, (os.cpu_count() or 4) - 2))")
L1ARG=()
if [ -n "$L1_HDF5" ]; then L1ARG=(--raw-l1-hdf5 "$L1_HDF5"); fi
stamp () { date '+%Y-%m-%d %H:%M:%S'; }

# 0. If Phase C is still running, wait for it (avoid fighting for the CPU).
while pgrep -f "run_phase_c.sh" > /dev/null; do
  echo "$(stamp) waiting for Phase C to finish..."; sleep 300
done

# 1. Tests. Any failure stops everything.
echo "$(stamp) [1/3] pytest"
if ! python -m pytest -q tests > "$LOG/pytest.log" 2>&1; then
  echo "$(stamp) STOP: tests failed, see $LOG/pytest.log" | tee "$OUT/STOPPED.txt"; exit 1
fi
tail -1 "$LOG/pytest.log"

# 2. Smoke run (about 10-20 min): catches crashes before the long run.
echo "$(stamp) [2/3] smoke run"
if ! python -u examples/gated_is_stage2.py --smoke --num-processes "$NPROC" ${L1ARG[@]+"${L1ARG[@]}"} \
      --outdir results/_smoke_stage2 > "$LOG/smoke.log" 2>&1; then
  echo "$(stamp) STOP: smoke run crashed, see $LOG/smoke.log" | tee "$OUT/STOPPED.txt"; exit 1
fi
if grep -q "Decision: STOP" results/_smoke_stage2/STAGE2_VERDICT.md 2>/dev/null; then
  cp results/_smoke_stage2/STAGE2_VERDICT.md "$OUT/STAGE2_VERDICT.md"
  echo "$(stamp) STOP: likelihood checks failed on real data" | tee "$OUT/STOPPED.txt"; exit 1
fi
grep -h "ms/sample" "$LOG/smoke.log" | tail -2

# 3. Full run. Resumable: rerunning skips finished cells.
echo "$(stamp) [3/3] full run with $NPROC processes"
python -u examples/gated_is_stage2.py --num-processes "$NPROC" ${L1ARG[@]+"${L1ARG[@]}"} --max-hours 10 \
  --outdir "$OUT" > "$LOG/full.log" 2>&1
echo "$(stamp) full run exited with code $?"
cat "$OUT/STAGE2_VERDICT.md" 2>/dev/null
echo "$(stamp) DONE" > "$OUT/DONE.txt"
