#!/usr/bin/env bash
# Phase C: final numbers for the audit paper.
# Gate recipe (fixed in advance from Phase B): band-limited 20-23 Hz, zero fill, half-width 0.1 s,
# whitened-excess localiser at 8 sigma. The true-time ("oracle") gate is reported beside it.
# Usage (repo root, conda env active):
#   export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
#   bash scripts/run_phase_c.sh /absolute/path/to/L-L1_GWOSC_4KHZ_R1-1187006835-4096.hdf5
set -u
L1_HDF5="${1:-}"
HALF=0.1
GATE_ENV="ADAPT_GATE_BAND_HZ=20,23 ADAPT_GATE_FILL=zero ADAPT_LOCALIZER=excess ADAPT_EXCESS_SIGMA=8"
LOG=results/phase_c_logs
mkdir -p "$LOG"

run () {  # run <outdir> <command...>   (GATE_ENV is applied to every run)
  local out="$1"; shift
  if [ -f "$out/.done" ]; then echo "[skip] $out"; return 0; fi
  mkdir -p "$out"
  echo "{\"env\": \"$GATE_ENV\", \"half_width_s\": $HALF}" > "$out/gate_env.json"
  echo "[run ] $out"
  env $GATE_ENV "$@" > "$LOG/$(basename "$out").log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$out/.done"; else echo "[FAIL] $out (exit $rc) see $LOG/$(basename "$out").log"; fi
  return 0
}

# C1. Localiser false-trigger audit (fast)
run results/localizer_audit_v3 python -u examples/localizer_audit.py --outdir results/localizer_audit_v3 --pad-s $HALF

# C2. Real Gravity Spy panel (localiser gate + oracle gate, both +-0.1 s)
run results/stress_real_glitches_v3_final python -u examples/stress_real_glitches.py --seed 0 --num-samples 512 \
  --wide-snr-panel --native-only --n-noise-cells 5 --selected-csv data/gravity_spy/selected_h1_o3_wide.csv \
  --force-gate-half-s $HALF --outdir results/stress_real_glitches_v3_final --figdir results/stress_real_glitches_v3_final

# C3. Loudness sweep (main grid, then the low-loudness extension into the same folder)
if [ ! -f results/collapse_threshold_v3/.done ]; then
  mkdir -p results/collapse_threshold_v3
  echo "[run ] results/collapse_threshold_v3"
  env $GATE_ENV python -u examples/collapse_threshold.py --seed 0 --num-samples 512 --n-seeds 5 \
      --force-gate-half-s $HALF --outdir results/collapse_threshold_v3 --figdir results/collapse_threshold_v3 \
      > "$LOG/collapse_threshold_v3.log" 2>&1 \
  && env $GATE_ENV python -u examples/collapse_threshold.py --seed 0 --num-samples 512 --extend-low-rho \
      --force-gate-half-s $HALF --outdir results/collapse_threshold_v3 --figdir results/collapse_threshold_v3 \
      >> "$LOG/collapse_threshold_v3.log" 2>&1 \
  && touch results/collapse_threshold_v3/.done \
  || echo "[FAIL] results/collapse_threshold_v3 see $LOG/collapse_threshold_v3.log"
fi

# C4. Unsubtracted GW170817 Livingston frame
if [ -n "$L1_HDF5" ]; then
  run results/gw170817_raw_l1_v3_final python -u examples/gw170817_raw_l1.py --num-processes 2 \
    --strain-hdf5 "$L1_HDF5" --force-gate-half-s $HALF --outdir results/gw170817_raw_l1_v3_final
else
  echo "[FAIL] raw L1 skipped: pass the GWOSC L1 HDF5 path as the first argument"
fi

# C5. Fifteen-parameter importance-sampled comparison on every collapsed real cell (slowest step)
if [ -f results/stress_real_glitches_v3_final/.done ]; then
  run results/multi_param_validation_v3 python -u examples/multi_param_validation.py --num-processes 2 \
    --skip-synthetic --all-morphologies --force-gate-half-s $HALF \
    --real-csv results/stress_real_glitches_v3_final/results.csv \
    --real-catalogue results/stress_real_glitches_v3_final/glitch_catalogue.csv \
    --outdir results/multi_param_validation_v3
else
  echo "[FAIL] multi-parameter step skipped: real panel did not finish"
fi

python -u examples/aggregate_phase_c.py | tee results/PHASE_C_REPORT.md
