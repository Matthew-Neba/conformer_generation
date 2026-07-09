#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd -- "$ROOT/.." && pwd)"
cd "$ROOT"

RDKIT_PREFIX="${RDKIT_PREFIX:-$REPO_ROOT/.micromamba/envs/rdkit-cpp}"
RDKIT_LIB="$RDKIT_PREFIX/lib"
RDKIT_CMAKE="$RDKIT_PREFIX/bin/cmake"
TORCH_NCCL_LIB="$(find "$ROOT/.venv" "$REPO_ROOT/venv" -path '*/nvidia/nccl/lib' -type d -print -quit 2>/dev/null || true)"
CONFORMER_RDKIT_LIB="$ROOT/build/libconformer_rdkit.so"
CONFORMER_RDKIT_BUILD_DIR="$ROOT/build/nvmolkit"
CONFORMER_CACHE_DIR="$ROOT/ocean/conformer_gen/cache/rdmol_v1"

if [ ! -f "$ROOT/config/conformer_gen.ini" ]; then
  echo "Missing config: $ROOT/config/conformer_gen.ini"
  exit 1
fi

ENV_NAME="${PUFFER_ENV_NAME:-conformer_gen}"
UV_REQS=(--with pybind11 --with numpy)
PUFFER_C_EXT="$(find "$ROOT/pufferlib" -maxdepth 1 -name '_C*.so' -print -quit)"
PUFFER_C_BACKUP="$(mktemp)"
PUFFER_C_HAD_BACKUP=0
if [ -n "$PUFFER_C_EXT" ] && [ -f "$PUFFER_C_EXT" ]; then
  cp "$PUFFER_C_EXT" "$PUFFER_C_BACKUP"
  PUFFER_C_HAD_BACKUP=1
fi

restore_puffer_backend() {
  if [ "$PUFFER_C_HAD_BACKUP" -eq 1 ] && [ -n "$PUFFER_C_EXT" ]; then
    cp "$PUFFER_C_BACKUP" "$PUFFER_C_EXT"
  elif [ -n "$PUFFER_C_EXT" ]; then
    rm -f "$PUFFER_C_EXT"
  fi
  rm -f "$PUFFER_C_BACKUP"
}
trap restore_puffer_backend EXIT

# 1) build puffer core backend for this env
unset DEBUG
EXTRA_CFLAGS="${EXTRA_CFLAGS:+$EXTRA_CFLAGS }-DSKIP_TERMINAL_MMFF_REWARD=1" \
  uv run "${UV_REQS[@]}" ./build.sh conformer_gen

# 2) rebuild the RDKit/nvMolKit shim so its exported chem_mol_* symbols match the env
if [ ! -f "$CONFORMER_RDKIT_BUILD_DIR/CMakeCache.txt" ]; then
  echo "Missing configured build dir: $CONFORMER_RDKIT_BUILD_DIR"
  echo "Expected a configured nvMolKit CMake tree with the conformer_rdkit target."
  exit 1
fi
if [ ! -x "$RDKIT_CMAKE" ]; then
  echo "Missing cmake binary: $RDKIT_CMAKE"
  exit 1
fi
"$RDKIT_CMAKE" --build "$CONFORMER_RDKIT_BUILD_DIR" --target conformer_rdkit
if [ ! -f "$CONFORMER_RDKIT_LIB" ]; then
  echo "Missing $CONFORMER_RDKIT_LIB after rebuild"
  exit 1
fi

# 3) preload shim + RDKit libs for import/eval runtime
export LD_LIBRARY_PATH="${TORCH_NCCL_LIB:+$TORCH_NCCL_LIB:}$RDKIT_LIB:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONFORMER_RDKIT_LIB${LD_PRELOAD:+:$LD_PRELOAD}"

uv run "${UV_REQS[@]}" python - <<'PY'
import pufferlib._C as C
print("Loaded _C for env:", getattr(C, "env_name", "<unknown>"))
PY

mkdir -p "$CONFORMER_CACHE_DIR"
cache_count="$(find "$CONFORMER_CACHE_DIR" -type f -name '*.rdmol' | wc -l)"
echo "Loaded cached RDKit molecule templates: $cache_count ($CONFORMER_CACHE_DIR)"

has_load_model_path=0
use_random_policy=0
filtered_args=()
for arg in "$@"; do
  if [ "$arg" = "--random" ]; then
    use_random_policy=1
    continue
  fi
  filtered_args+=("$arg")
  if [ "$arg" = "--load-model-path" ] || [[ "$arg" == --load-model-path=* ]]; then
    has_load_model_path=1
  fi
done

eval_args=("${filtered_args[@]}")
if [ "$use_random_policy" -eq 0 ] && [ "$has_load_model_path" -eq 0 ]; then
  checkpoint_root="$ROOT/checkpoints/$ENV_NAME"
  latest_model="$(
    find "$checkpoint_root" -type f -name '*.bin' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print }'
  )"
  if [ -z "$latest_model" ]; then
    echo "No checkpoints found in $checkpoint_root"
    exit 1
  fi
  echo "Using latest checkpoint: $latest_model"
  eval_args=(--load-model-path "$latest_model" "${eval_args[@]}")
fi

if [ "$use_random_policy" -eq 1 ]; then
  echo "Using random policy (no model checkpoint loaded)"
fi

# 4) start eval
uv run "${UV_REQS[@]}" puffer eval "$ENV_NAME" "${eval_args[@]}"
