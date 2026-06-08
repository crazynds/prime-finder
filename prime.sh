#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
DATA_DIR="$SCRIPT_DIR/data"

# ---------- configurable via environment ----------
SMALL_PRIMES_COUNT="${SMALL_PRIMES_COUNT:-10000}"
GPU_PRIMES_COUNT="${GPU_PRIMES_COUNT:-100000000}"
A_MIN="${A_MIN:-50000}"
A_MAX="${A_MAX:-51000}"
C_MIN="${C_MIN:-100}"
MPI_NP="${MPI_NP:-$(nproc)}"
CPU_PHASE2_ONLY="${CPU_PHASE2_ONLY:-}"   # set to "--cpu-phase2-only" to enable

SMALL_PRIMES_BIN="$DATA_DIR/small_primes.bin"
GPU_PRIMES_BIN="$DATA_DIR/gpu_primes.bin"
# --------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  build         Build all binaries from source (clean CMake build)
  gen_primes    Generate prime tables into data/
  run           Run the distributed prime hunter (blocking)
  test          Run all tests
  bench         Run Miller-Rabin benchmark
  bench-gpu     Run GPU trial-division benchmark (requires gen_primes first)
  clean         Remove all generated data (primes, results, checkpoints)
  monitor       Watch the live monitor output

Environment variables (override defaults):
  SMALL_PRIMES_COUNT   Primes for sieve table        (default: $SMALL_PRIMES_COUNT)
  GPU_PRIMES_COUNT     Primes for GPU trial division  (default: $GPU_PRIMES_COUNT)
  A_MIN                Exponent range start           (default: $A_MIN)
  A_MAX                Exponent range end             (default: $A_MAX)
  C_MIN                Minimum c value                (default: $C_MIN)
  MPI_NP               Number of MPI processes        (default: $MPI_NP)
  CPU_PHASE2_ONLY      Set to --cpu-phase2-only        (default: off)
EOF
}

cmd_build() {
    echo "[build] Configuring..."
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    echo "[build] Compiling..."
    cmake --build "$BUILD_DIR" --parallel "$(nproc)"
    echo "[build] Done. Binaries in $BUILD_DIR/"
}

cmd_gen_primes() {
    mkdir -p "$DATA_DIR"
    local gen="$BUILD_DIR/gen_primes"
    if [[ ! -x "$gen" ]]; then
        echo "[gen_primes] Binary not found — run 'build' first."
        exit 1
    fi

    echo "[gen_primes] Generating small primes ($SMALL_PRIMES_COUNT) → $SMALL_PRIMES_BIN"
    "$gen" "$SMALL_PRIMES_COUNT" "$SMALL_PRIMES_BIN"

    echo "[gen_primes] Generating GPU primes ($GPU_PRIMES_COUNT) → $GPU_PRIMES_BIN"
    "$gen" "$GPU_PRIMES_COUNT" "$GPU_PRIMES_BIN"

    echo "[gen_primes] Done."
}

cmd_run() {
    local hunter="$BUILD_DIR/prime_hunter"
    if [[ ! -x "$hunter" ]]; then
        echo "[run] Binary not found — run 'build' first."
        exit 1
    fi
    if [[ ! -f "$SMALL_PRIMES_BIN" || ! -f "$GPU_PRIMES_BIN" ]]; then
        echo "[run] Prime tables not found — run 'gen_primes' first."
        exit 1
    fi

    mkdir -p "$DATA_DIR"

    echo "[run] mpirun -np $MPI_NP prime_hunter  a=[$A_MIN..$A_MAX] c_min=$C_MIN"
    mpirun -np "$MPI_NP" \
        --wdir "$DATA_DIR" \
        "$hunter" \
        "$SMALL_PRIMES_BIN" \
        "$GPU_PRIMES_BIN" \
        "$A_MIN" "$A_MAX" "$C_MIN" \
        ${CPU_PHASE2_ONLY}
}

cmd_test() {
    local tests="$BUILD_DIR/prime_tests"
    if [[ ! -x "$tests" ]]; then
        echo "[test] Binary not found — run 'build' first."
        exit 1
    fi
    "$tests" --test
}

cmd_bench() {
    local tests="$BUILD_DIR/prime_tests"
    if [[ ! -x "$tests" ]]; then
        echo "[bench] Binary not found — run 'build' first."
        exit 1
    fi
    "$tests" --bench
}

cmd_bench_gpu() {
    local tests="$BUILD_DIR/prime_tests"
    if [[ ! -x "$tests" ]]; then
        echo "[bench-gpu] Binary not found — run 'build' first."
        exit 1
    fi
    if [[ ! -f "$GPU_PRIMES_BIN" ]]; then
        echo "[bench-gpu] Prime table not found — run 'gen_primes' first."
        exit 1
    fi
    "$tests" --bench-gpu "$GPU_PRIMES_BIN"
}

cmd_clean() {
    echo "[clean] Removing data files in $DATA_DIR/ ..."
    rm -f "$DATA_DIR"/phase1_survivors.txt \
          "$DATA_DIR"/phase2_primes.txt \
          "$DATA_DIR"/monitor.txt \
          "$DATA_DIR"/checkpoint.txt \
          "$DATA_DIR"/checkpoint_phase2.txt \
          "$DATA_DIR"/small_primes.bin \
          "$DATA_DIR"/gpu_primes.bin
    if [[ -d "$DATA_DIR" ]]; then
        rmdir --ignore-fail-on-non-empty "$DATA_DIR"
    fi
    echo "[clean] Done."
}

cmd_monitor() {
    local mon="$DATA_DIR/monitor.txt"
    if [[ ! -f "$mon" ]]; then
        echo "[monitor] $mon not found — is the program running?"
        exit 1
    fi
    watch -n1 cat "$mon"
}

# ---------- dispatch ----------
if [[ $# -lt 1 ]]; then usage; exit 1; fi

case "$1" in
    build)      cmd_build ;;
    gen_primes) cmd_gen_primes ;;
    run)        cmd_run ;;
    test)       cmd_test ;;
    bench)      cmd_bench ;;
    bench-gpu)  cmd_bench_gpu ;;
    clean)      cmd_clean ;;
    monitor)    cmd_monitor ;;
    *)          echo "Unknown command: $1"; echo; usage; exit 1 ;;
esac
