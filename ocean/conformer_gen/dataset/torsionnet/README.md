# TorsionNet Datasets

Imported from `torsionnet_paper_version/TorsionNet`.

The original JSON keys are normalized as:

- `standard` -> `E0` after converting back into raw energy units
- `total` -> `Z0`
- `temp_normal` -> `tau = 1 / temp_normal`

Scoring should follow the local TorsionNet environment code:

```text
contribution = exp(-(energy - E0) / tau) / Z0
```

This is the same score as TorsionNet's `energy * temp_normal - standard` form, just written with division by `tau`.

Temperature values:

- `alkane_*`: `tau = 1.0`
- `lignin_train`: `tau = 4.0`
- `lignin_validation_8`: `tau = 4.0`
- `lignin_test_sgmd_8`: `tau = 3.969829297340214`

Directories:

- `alkane_all`: all 10,000 generated branched alkanes from `huge_hc_set.zip`
- `alkane_train`: the paper training subset, rbn 7 through 10, 4,332 records
- `alkane_validation_10`: single unseen 10-rbn validation molecule
- `alkane_test_11`: 11-rbn test molecule
- `alkane_test_22`: 22-rbn test molecule
- `lignin_train`: 12 lignin train molecules, 2 through 7 monomer units
- `lignin_validation_8`: single validation 8-lignin
- `lignin_test_sgmd_8`: final SGMD-normalized 8-lignin test molecule

For lignin, `.mol` files are copied beside their JSON records and referenced by the `molfile` field.
