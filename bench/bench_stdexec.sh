#!/usr/bin/env bash
# Benchmark LAMMPS (KOKKOS package) with and without kokkos-stdexec.
#
# Two executables are built from the SAME source tree. The only difference is
# -DEXTERNAL_KOKKOS_STDEXEC=ON, which defines LMP_KOKKOS_STDEXEC and switches
# VerletKokkos to the sender-based force clear that overlaps the halo exchange.
#
#   build-orig/lmp      original Kokkos Verlet integrator
#   build-stdexec/lmp   kokkos-stdexec Verlet integrator
#
# Usage:
#   ./bench/bench_stdexec.sh build [orig|stdexec|both]   configure + compile (default: both)
#   ./bench/bench_stdexec.sh run                         benchmark both variants, print summary
#   ./bench/bench_stdexec.sh all                         build + run
#   ./bench/bench_stdexec.sh summary <results-dir>       re-print summary of an earlier run
#
# Everything is configurable through the environment, e.g.
#   SIZES="10 20 40" STEPS=2000 REPS=10 ./bench/bench_stdexec.sh run
#   LAUNCHER="srun -n 1 --gpus=1" ./bench/bench_stdexec.sh run
#
# Defaults to 4 MPI ranks, one GPU each (RANKS=4). Run `run` on the GPU node. Builds and runs need CUDA, the machine this
# was written on has no GPU.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMMPS_ROOT="${LAMMPS_ROOT:-$(cd "$HERE/.." && pwd)}"

# ---- installs ---------------------------------------------------------------
KOKKOS_ROOT="${KOKKOS_ROOT:-$HOME/Kokkos/kokkos/install}"
KOKKOS_STDEXEC_ROOT="${KOKKOS_STDEXEC_ROOT:-$HOME/RAndD/kokkos-stdexec/install}"
STDEXEC_ROOT="${STDEXEC_ROOT:-$HOME/RAndD/stdexec/install}"

# ---- toolchain (matches the paper: clang 22.1.8, CUDA 12.8, OpenMPI) --------
CXX="${CXX:-$HOME/software/spack/opt/spack/linux-zen4/llvm-22.1.8-fppfoe4otpl3stueupo64nmhbfuq6do6/bin/clang++}"
CC="${CC:-${CXX%++}}"
MPI_CXX="${MPI_CXX:-/opt/spack/rhel9_x86/stack-2025-03/spack/opt/spack/linux-rhel9-x86_64/gcc-11.5.0/openmpi-4.1.7-nblrmap7qoogftg4ko5xpozktpsyqgfi/bin/mpicxx}"
RUNTIME_LD_PATH="${RUNTIME_LD_PATH:-$HOME/software/spack/opt/spack/linux-zen4/llvm-22.1.8-fppfoe4otpl3stueupo64nmhbfuq6do6/lib/x86_64-unknown-linux-gnu:/opt/spack/rhel9_x86/stack-2025-03/spack/opt/spack/linux-rhel9-x86_64/gcc-11.5.0/gcc-13.3.0-idfi4sskipw5bxzgpl4bc4qcvssrtqlr/lib64}"
JOBS="${JOBS:-$(nproc)}"

# ---- build dirs -------------------------------------------------------------
BUILD_ORIG="${BUILD_ORIG:-$LAMMPS_ROOT/build-orig}"
BUILD_STDEXEC="${BUILD_STDEXEC:-$LAMMPS_ROOT/build-stdexec}"

# ---- benchmark parameters ---------------------------------------------------
# SIZES are fcc unit cells per dimension, atoms = 4*n^3.
#   10 -> 4k, 20 -> 32k, 40 -> 256k, 60 -> 864k atoms
# Small systems are launch-latency bound, which is where overlapping the force
# clear with communication has a chance to show up. Large systems are kernel
# bound and should show no difference.
SIZES="${SIZES:-10 20 40 60}"
STEPS="${STEPS:-1000}"      # timed steps
WARM="${WARM:-100}"         # untimed warm-up steps in the same process
REPS="${REPS:-5}"           # repetitions per (variant, size)
RANKS="${RANKS:-4}"         # MPI ranks on the node, one GPU each
NGPU="${NGPU:-$RANKS}"      # GPUs per node passed to `-k on g`
MPIRUN="$(command -v mpirun || echo "$(dirname "$MPI_CXX")/mpirun")"
LAUNCHER="${LAUNCHER-$MPIRUN -np $RANKS}"   # set LAUNCHER="" to exec directly, or e.g. "srun -n 4 --gpus-per-task=1"
KK_ARGS="${KK_ARGS:--k on g $NGPU -sf kk}"
# Extra LAMMPS args appended to every run.
EXTRA_ARGS="${EXTRA_ARGS:-}"
# Package configurations, "label:args" separated by ';'. They matter:
#  - gpu-default: GPU default is `neigh full newton off`. The ORIGINAL code
#    fuses the force zero into the pair kernel here, kokkos-stdexec does not.
#  - gpu-nofuse: same, but the token NOFUSE sets LMP_KOKKOS_NO_FUSE_FORCE_CLEAR=1
#    so the original also launches the zero as a separate kernel. Both variants
#    then run the same kernels and only the overlap differs. This is the fair
#    comparison for the paper.
#  - half-newton: no fusion in either variant (fusion needs a full list).
CONFIGS="${CONFIGS:-gpu-nofuse:NOFUSE;gpu-default:;half-newton:-pk kokkos newton on neigh half}"
INPUT="${INPUT:-$HERE/in.lj_stdexec_bench}"
OUTDIR="${OUTDIR:-$HERE/results/$(date +%Y%m%d-%H%M%S)}"

die() { echo "error: $*" >&2; exit 1; }

# ---- build ------------------------------------------------------------------
configure_and_build() {
  local variant="$1" bdir stdexec_flag=()
  case "$variant" in
    orig)    bdir="$BUILD_ORIG" ;;
    stdexec) bdir="$BUILD_STDEXEC"
             stdexec_flag=(-D EXTERNAL_KOKKOS_STDEXEC=ON
                           -D KokkosStdexec_ROOT="$KOKKOS_STDEXEC_ROOT"
                           -D stdexec_ROOT="$STDEXEC_ROOT") ;;
    *) die "unknown variant $variant" ;;
  esac

  echo "=== building $variant in $bdir"

  # The host compiler must be the one Kokkos was installed with. An install
  # made with nvcc_wrapper (ID NVIDIA, e.g. install-hopper) routes every compile
  # through nvcc_wrapper, whose host compiler is that gcc. Handing it clang-only
  # flags such as -fopenmp=libomp then fails with "c++: unrecognized option".
  local kk_common kk_id kk_cxx
  kk_common="$(ls "$KOKKOS_ROOT"/lib*/cmake/Kokkos/KokkosConfigCommon.cmake 2>/dev/null | head -n1)"
  [[ -n "$kk_common" ]] || die "no Kokkos cmake config under $KOKKOS_ROOT"
  kk_id="$(sed -n 's/^set(Kokkos_CXX_COMPILER_ID "\(.*\)")/\1/p' "$kk_common")"
  kk_cxx="$(sed -n 's/^set(Kokkos_CXX_COMPILER "\(.*\)")/\1/p' "$kk_common")"
  local cxx="$CXX" cc="$CC"
  if [[ "$kk_id" == "NVIDIA" ]]; then
    echo "note: Kokkos at $KOKKOS_ROOT is an nvcc_wrapper install, host compiler $kk_cxx; using it for LAMMPS"
    cxx="$kk_cxx"; cc="${kk_cxx%c++}cc"; [[ -x "$cc" ]] || cc="${kk_cxx%++}"
    export NVCC_WRAPPER_DEFAULT_COMPILER="$kk_cxx"
  else
    export NVCC_WRAPPER_DEFAULT_COMPILER="$CXX"
  fi
  # A reused build dir can carry a launcher from another Kokkos install.
  if [[ -f "$bdir/CMakeCache.txt" ]] &&
     ! grep -q "^Kokkos_COMPILE_LAUNCHER:.*=$KOKKOS_ROOT/" "$bdir/CMakeCache.txt"; then
    echo "stale CMake cache in $bdir (different Kokkos install), resetting it"
    rm -rf "$bdir/CMakeCache.txt" "$bdir/CMakeFiles"
  fi
  local mpi_flag=()
  [[ -x "$MPI_CXX" ]] && mpi_flag=(-D MPI_CXX_COMPILER="$MPI_CXX")

  # Identical flags for both variants, except the stdexec ones.
  cmake -S "$LAMMPS_ROOT/cmake" -B "$bdir" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER="$cxx" -D CMAKE_C_COMPILER="$cc" \
    -D CMAKE_CXX_STANDARD=20 \
    -D PKG_KOKKOS=ON -D EXTERNAL_KOKKOS=ON \
    -D Kokkos_ROOT="$KOKKOS_ROOT" \
    -D Kokkos_DIR="$(dirname "$kk_common")" \
    -D BUILD_MPI=ON -D BUILD_OMP=OFF \
    -U OpenMP_C_FLAGS -U OpenMP_CXX_FLAGS \
    -D Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
    "${mpi_flag[@]}" "${stdexec_flag[@]}"
  cmake --build "$bdir" -j "$JOBS" --target lmp
}

do_build() {
  case "${1:-both}" in
    orig)    configure_and_build orig ;;
    stdexec) configure_and_build stdexec ;;
    both)    configure_and_build orig; configure_and_build stdexec ;;
    *) die "build: expected orig|stdexec|both" ;;
  esac
}

# ---- run --------------------------------------------------------------------
lmp_bin() { [[ "$1" == orig ]] && echo "$BUILD_ORIG/lmp" || echo "$BUILD_STDEXEC/lmp"; }

run_one() {
  local variant="$1" n="$2" rep="$3" cfg="$4" cfg_args="$5"
  local log="$OUTDIR/logs/${variant}_${cfg}_n${n}_r${rep}.log"
  local nofuse=0
  if [[ " $cfg_args " == *" NOFUSE "* ]]; then nofuse=1; cfg_args="${cfg_args//NOFUSE/}"; fi
  # shellcheck disable=SC2086
  LD_LIBRARY_PATH="$RUNTIME_LD_PATH:${LD_LIBRARY_PATH:-}" \
  LMP_KOKKOS_NO_FUSE_FORCE_CLEAR="$nofuse" \
    $LAUNCHER "$(lmp_bin "$variant")" $KK_ARGS $cfg_args $EXTRA_ARGS \
      -var n "$n" -var steps "$STEPS" -var warm "$WARM" \
      -in "$INPUT" -log none > "$log" 2>&1 \
    || { tail -n 20 "$log" >&2; die "$variant n=$n rep=$rep failed, see $log"; }

  # The banner proves which integrator actually ran.
  if [[ "$variant" == stdexec ]]; then
    grep -q "Kokkos stdexec:" "$log" || die "$log: stdexec banner missing, wrong binary?"
  else
    grep -q "Kokkos stdexec:" "$log" && die "$log: orig binary printed the stdexec banner"
  fi

  # Use the LAST Loop time, the first one belongs to the warm-up run.
  local line atoms loop
  line="$(grep "^Loop time of" "$log" | tail -n 1)"
  [[ -n "$line" ]] || die "$log: no 'Loop time' line"
  loop="$(awk '{print $4}' <<<"$line")"
  atoms="$(awk '{print $(NF-1)}' <<<"$line")"
  awk -v c="$cfg" -v v="$variant" -v n="$n" -v r="$rep" -v a="$atoms" -v t="$loop" -v s="$STEPS" \
    'BEGIN{printf "%s,%s,%d,%d,%d,%.6f,%.3f,%.6e\n", c, v, n, a, r, t, s/t, a*s/t}' \
    >> "$OUTDIR/results.csv"
  printf "  %-12s %-8s n=%-3s rep=%s  loop=%ss  atoms=%s\n" "$cfg" "$variant" "$n" "$rep" "$loop" "$atoms"
}

do_run() {
  for v in orig stdexec; do
    [[ -x "$(lmp_bin "$v")" ]] || die "$(lmp_bin "$v") missing, run: $0 build"
  done
  [[ -f "$INPUT" ]] || die "input $INPUT not found"
  mkdir -p "$OUTDIR/logs"
  echo "config,variant,n,atoms,rep,loop_s,steps_per_s,atom_steps_per_s" > "$OUTDIR/results.csv"

  {
    echo "date:     $(date -Is)"
    echo "host:     $(hostname)"
    echo "git:      $(git -C "$LAMMPS_ROOT" rev-parse --short HEAD) (dirty: $(git -C "$LAMMPS_ROOT" status --porcelain --untracked-files=no | wc -l) files)"
    echo "sizes:    $SIZES  steps=$STEPS warm=$WARM reps=$REPS"
    echo "args:     $LAUNCHER lmp $KK_ARGS <config> $EXTRA_ARGS"
    echo "configs:  $CONFIGS"
    echo "ranks:    $RANKS  (gpus/node $NGPU)"
    echo "kokkos:   $KOKKOS_ROOT"
    echo "stdexec:  $STDEXEC_ROOT"
    if command -v nvidia-smi >/dev/null; then
      nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader
    fi
  } | tee "$OUTDIR/info.txt"

  local IFS_SAVE="$IFS" entry cfg cfg_args
  IFS=';' read -ra cfg_list <<<"$CONFIGS"
  IFS="$IFS_SAVE"
  for entry in "${cfg_list[@]}"; do
    cfg="${entry%%:*}"; cfg_args="${entry#*:}"
    for n in $SIZES; do
      echo "--- $cfg  n=$n ($((4*n*n*n)) atoms total, $RANKS ranks)"
      for rep in $(seq 1 "$REPS"); do
        # Alternate the order so slow GPU clock/thermal drift hits both equally.
        if (( rep % 2 )); then order="orig stdexec"; else order="stdexec orig"; fi
        for v in $order; do run_one "$v" "$n" "$rep" "$cfg" "$cfg_args"; done
      done
    done
  done

  echo
  echo "results: $OUTDIR/results.csv"
  summarize "$OUTDIR"
}

# ---- summary ----------------------------------------------------------------
summarize() {
  command -v python3 >/dev/null || { echo "python3 not found, see $1/results.csv"; return; }
  python3 - "$1/results.csv" <<'EOF' | tee "$1/summary.txt"
import csv, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1])))
data = {}
for r in rows:
    data.setdefault((r["config"], int(r["n"]), r["variant"]), []).append(float(r["loop_s"]))
atoms = {int(r["n"]): int(r["atoms"]) for r in rows}
def ms(v): return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
print(f'{"config":<12} {"atoms":>9} {"orig [s]":>18} {"stdexec [s]":>18} {"speedup":>9}  (speedup = orig/stdexec, >1 means stdexec is faster)')
for cfg, n in sorted({(k[0], k[1]) for k in data}, key=lambda t: (t[0], t[1])):
    o, s = data.get((cfg, n, "orig")), data.get((cfg, n, "stdexec"))
    if not o or not s: continue
    (om, osd), (sm, ssd) = ms(o), ms(s)
    sp = om / sm
    # crude uncertainty of the ratio from the relative stddevs
    err = sp * ((osd/om)**2 + (ssd/sm)**2) ** 0.5
    print(f'{cfg:<12} {atoms[n]:>9} {om:>10.4f}±{osd:<7.4f} {sm:>10.4f}±{ssd:<7.4f} {sp:>6.3f}±{err:.3f}')
print("\nIf |speedup-1| is within the ± band, the two variants are indistinguishable.")
EOF
}

# ---- main -------------------------------------------------------------------
case "${1:-}" in
  build)   do_build "${2:-both}" ;;
  run)     do_run ;;
  all)     do_build both; do_run ;;
  summary) [[ -f "${2:-}/results.csv" ]] || die "usage: $0 summary <results-dir>"; summarize "$2" ;;
  *) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
