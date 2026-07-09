# Conformer Gen Architecture

## Modeling Assumptions

1. Torsion actions use heavy atoms only; hydrogens remain inside RDKit for geometry updates.
2. The action space rotates non-ring single bonds only. Double, triple, and ring bonds are not action-space torsions.
3. Torsion edits use a rigid-rotor approximation.
4. Torsion actions choose one of six 60-degree buckets directly.
5. Ring rotatable bonds are excluded from the action space because ring torsions are coupled; changing one can force changes across other ring torsions.
6. TFD is matched to RDKit/TorsionNet's unweighted default behavior: `useWeights=False`, `maxDev="equal"` for non-ring torsions.

## Molecule Loading

Training loads only selected `*_train` molecules: alkanes, lignins, or both.

Each dataset JSON is converted to one preinitialized RDKit template molecule. Templates are cached as RDKit pickles under `ocean/conformer_gen/cache/rdmol_v1` and reused across processes when newer than the source JSON/molfile.

On reset, the env picks a random loaded molecule, deep-clones its cached RDKit template, and starts from that cached initial conformer.

## Observations

Observations are fixed-size padded arrays for one molecule at a time.

Global features:
- normalized heavy atom count
- normalized non-ring rotatable torsion count
- normalized proposal count
- normalized accepted conformer count

Per heavy atom:
- valid atom slot mask
- atomic number normalized by `MAX_ATOMIC_NUMBER`
- x coordinate
- y coordinate
- z coordinate

Coordinates are centered by subtracting the heavy-atom mean position and then divided by `COORD_SCALE`. They are not rotation-normalized.

Per torsion:
- valid torsion slot mask
- four heavy-atom feature blocks for the torsion atoms
- sin/cos of the current torsion bucket

Each torsion atom feature block contains:
- atomic number normalized by `MAX_ATOMIC_NUMBER`
- centered/scaled x coordinate
- centered/scaled y coordinate
- centered/scaled z coordinate

## Actions

The action space contains:
- six bucket actions for every padded non-ring torsion slot
- one proposal action

Valid torsion edits update the current RDKit conformer and receive `DEFAULT_EDIT_STEP_PENALTY`. Torsion edits into padded slots receive `DEFAULT_OUT_OF_BOUNDS_EDIT_PENALTY`.

Proposal actions archive conformers whose torsion bucket state has not already been accepted in the current episode. Duplicate proposals receive `DUPLICATE_PROPOSE_PENALTY`.

## Rewards

Valid torsion edits return `DEFAULT_EDIT_STEP_PENALTY`. Torsion edits into padded slots receive `DEFAULT_OUT_OF_BOUNDS_EDIT_PENALTY`, and duplicate proposals receive `DUPLICATE_PROPOSE_PENALTY`.

At episode termination, all accepted conformers are MMFF-minimized with the same RDKit MMFF path used by eval. The terminal reward is the sum of the accepted conformers' normalized Gibbs masses:

```c
gibbs = exp(-((mmff_energy - E0) / tau)) / Z0
```

This makes training sparse, but uses the same post-minimization objective used for eval rather than a geometry proxy. The reward uses Gibbs mass instead of log-Gibbs so additional good conformers add positive reward instead of making the episode return more negative.

## Rendering

Rendering uses the current RDKit molecule topology and coordinates. Render-only diagnostics minimize temporary conformer copies with MMFF and display current/proposed/accepted Gibbs summaries.
