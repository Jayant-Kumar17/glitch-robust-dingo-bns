#!/usr/bin/env bash
# Phase B: the decisive honest-gate experiments after the packaging-convention fix.
# Usage (from the repo root, inside the conda env):
#   export PYTHONPATH=DINGO-BNS/dingo:src:examples KMP_DUPLICATE_LIB_OK=TRUE
#   bash scripts/run_phase_b.sh [/path/to/L-L1_GWOSC_4KHZ_R1-1187006835-4096.hdf5]
# Every run writes to a NEW results/*_v3_* folder. Nothing existing is overwritten.
# A step that fails is logged and the script moves on; finished folders are skipped on rerun.
set -u
L1_HDF5="${1:-}"
BAND="20,23"
LOG=results/phase_b_logs
mkdir -p "$LOG"

run () {  # run <outdir> <env assignments or "-"> <command...>
  local out="$1"; local envs="$2"; shift 2
  if [ -f "$out/.done" ]; then echo "[skip] $out"; return 0; fi
  mkdir -p "$out"
  printf '{"ADAPT_GATE_BAND_HZ": "%s", "ADAPT_GATE_FILL": "%s"}\n' \
    "$(echo "$envs" | sed -n 's/.*ADAPT_GATE_BAND_HZ=\([^ ]*\).*/\1/p')" \
    "$(echo "$envs" | sed -n 's/.*ADAPT_GATE_FILL=\([^ ]*\).*/\1/p')" > "$out/gate_env.json"
  echo "[run ] $out  ($envs)"
  if [ "$envs" = "-" ]; then
    "$@" > "$LOG/$(basename "$out").log" 2>&1
  else
    env $envs "$@" > "$LOG/$(basename "$out").log" 2>&1
  fi
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$out/.done"; else echo "[FAIL] $out (exit $rc) see $LOG/$(basename "$out").log"; fi
  return 0
}

# --- B1. Forced gates on CLEAN GW170817: what gate does the network tolerate? ---------------
GT=results/gate_tolerance_v3
for half in 0.05 0.1 0.2 0.4 0.8; do
  for variant in raw_zero band_zero band_noise; do
    case $variant in
      raw_zero)   extra="--fill zero" ;;
      band_zero)  extra="--fill zero --gate-band 20 23" ;;
      band_noise) extra="--fill noise --gate-band 20 23" ;;
    esac
    out="$GT/near_${variant}_h${half}"
    run "$out" - python -u examples/clean_gate_control.py --seed 0 --num-samples 512 \
      --gate-half-s "$half" --t-rel-min -2.0 --t-rel-max -0.2 $extra --outdir "$out" --figdir "$out"
  done
done
for half in 0.1 0.4 0.8; do
  for variant in band_zero band_noise; do
    case $variant in
      band_zero)  extra="--fill zero --gate-band 20 23" ;;
      band_noise) extra="--fill noise --gate-band 20 23" ;;
    esac
    out="$GT/far_${variant}_h${half}"
    run "$out" - python -u examples/clean_gate_control.py --seed 0 --num-samples 512 \
      --gate-half-s "$half" --t-rel-min -60.0 --t-rel-max -5.0 $extra --outdir "$out" --figdir "$out"
  done
done

# --- B2. The 16 hardening cells with detector-driven gates ----------------------------------
MH=results/journal_method_hardening_v3
run "${MH}_raw_zero_h0.4" - python -u examples/method_hardening.py --profile v2 --num-samples 512 \
  --batch-size 512 --skip-synth-oracle --overwrite --outdir "${MH}_raw_zero_h0.4"
for half in 0.4 0.1; do
  run "${MH}_band_zero_h${half}" "ADAPT_GATE_BAND_HZ=$BAND" python -u examples/method_hardening.py \
    --profile v2 --num-samples 512 --batch-size 512 --skip-synth-oracle --overwrite \
    --force-gate-half-s "$half" --arms adapt_full poison_welch glitch_orig_asd --outdir "${MH}_band_zero_h${half}"
  run "${MH}_band_noise_h${half}" "ADAPT_GATE_BAND_HZ=$BAND ADAPT_GATE_FILL=noise" python -u examples/method_hardening.py \
    --profile v2 --num-samples 512 --batch-size 512 --skip-synth-oracle --overwrite \
    --force-gate-half-s "$half" --arms adapt_full poison_welch glitch_orig_asd --outdir "${MH}_band_noise_h${half}"
done

# --- B3. Real Gravity Spy panel (detector gate and centred oracle gate) ---------------------
RG=results/stress_real_glitches_v3
REAL="python -u examples/stress_real_glitches.py --seed 0 --num-samples 512 --wide-snr-panel --native-only --n-noise-cells 5 --selected-csv data/gravity_spy/selected_h1_o3_wide.csv"
for half in 0.1 0.4; do
  run "${RG}_band_noise_h${half}" "ADAPT_GATE_BAND_HZ=$BAND ADAPT_GATE_FILL=noise" $REAL \
    --force-gate-half-s "$half" --outdir "${RG}_band_noise_h${half}" --figdir "${RG}_band_noise_h${half}"
  run "${RG}_band_zero_h${half}" "ADAPT_GATE_BAND_HZ=$BAND" $REAL \
    --force-gate-half-s "$half" --outdir "${RG}_band_zero_h${half}" --figdir "${RG}_band_zero_h${half}"
done

# --- B4. Unsubtracted GW170817 L1 ------------------------------------------------------------
if [ -n "$L1_HDF5" ]; then
  RL=results/gw170817_raw_l1_v3
  run "${RL}_band_zero" "ADAPT_GATE_BAND_HZ=$BAND" python -u examples/gw170817_raw_l1.py \
    --num-processes 2 --strain-hdf5 "$L1_HDF5" --outdir "${RL}_band_zero"
  run "${RL}_band_noise" "ADAPT_GATE_BAND_HZ=$BAND ADAPT_GATE_FILL=noise" python -u examples/gw170817_raw_l1.py \
    --num-processes 2 --strain-hdf5 "$L1_HDF5" --outdir "${RL}_band_noise"
else
  echo "[skip] raw L1: pass the GWOSC L1 HDF5 path as the first argument to run it"
fi

python -u examples/aggregate_phase_b.py | tee results/PHASE_B_REPORT.md
