# Conformer Generation

RL-based molecular conformer generation built as a custom PufferLib environment.

The agent edits molecular torsions, proposes conformer sets, and scores accepted conformers with MMFF/Gibbs-style rewards.

## Stack

- **PufferLib** for RL training, vectorized environments, sweeps, and rendering hooks
- **RDKit** for molecule parsing, torsion edits, conformer storage, and MMFF scoring
- **nvMolKit** for batched GPU MMFF minimization
- **CUDA/PyTorch** for GPU training

## Key Files

- `ocean/conformer_gen/`: environment, RDKit wrapper, and bootstrap scripts
- `config/conformer_gen.ini`: training configuration
- `pufferlib/`: PufferLib runtime/training code

## Run

```bash
./ocean/conformer_gen/bootstrap_train.sh
```

```bash
./ocean/conformer_gen/bootstrap_eval.sh
```

Checkpoints, logs, virtualenvs, build outputs, and conformer caches are intentionally not tracked.
