#ifndef CFFIWRAPPER_H
#define CFFIWRAPPER_H

#ifndef __cplusplus
#include <stdbool.h>
#endif

// need this guard for proper linking, c++ name mangles the signature of functions, c does not expect that.
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ChemMol ChemMol;

bool enable_logging(void);

bool chem_mol_from_smiles(const char* smiles, ChemMol** out_mol);
bool chem_mol_from_molfile(const char* path, ChemMol** out_mol);
bool chem_mol_from_pickle(const char* path, ChemMol** out_mol);
bool chem_mol_to_pickle(const ChemMol* mol, const char* path);
bool chem_mol_clone(const ChemMol* src, ChemMol** out_mol);
bool chem_mol_clone_conformer(const ChemMol* src, int conf_id, ChemMol** out_mol);
bool chem_mol_free(ChemMol* mol);

bool chem_mol_num_heavy_atoms(const ChemMol* mol, int* out_num_heavy_atoms);
bool chem_mol_num_nonring_rotatable_torsions(const ChemMol* mol, int* out_num_torsions);
bool chem_mol_get_atomic_numbers(const ChemMol* mol, int* out_atomic_numbers, int max_atoms, int* out_written);

// Returns false only when RDKit cannot embed the molecule for this seed; callers may retry with another seed.
bool chem_mol_init_conformer(ChemMol* mol, unsigned int seed, int* out_conf_id);
bool chem_mol_copy_conformer(ChemMol* mol, int* out_conf_id);
bool chem_mol_remove_conformer(ChemMol* mol, int conf_id);
bool chem_mol_set_active_conformer(ChemMol* mol, int conf_id);
bool chem_mol_get_active_conformer_id(const ChemMol* mol, int* out_conf_id);
bool chem_mol_set_torsion_bucket(ChemMol* mol, int torsion_idx, int bucket);
bool chem_mol_get_torsion_bucket(const ChemMol* mol, int torsion_idx, int* out_bucket);
bool chem_mol_minimize_mmff_conf(ChemMol* mol, int conf_id, int max_iters);
bool chem_mol_minimize_mmff_all_conformers(
        ChemMol* mol,
        int max_iters,
        const int* conf_ids,
        int num_conf_ids,
        double* out_energies);
bool chem_mol_energy_mmff(ChemMol* mol, int conf_id, double* out_energy);

bool chem_mol_get_torsion_atoms(const ChemMol* mol, int torsion_idx, int* out_atoms4);
bool chem_mol_tfd_between_conformers(const ChemMol* mol, int conf_id_a, int conf_id_b, double* out_tfd);

bool chem_mol_get_coords(const ChemMol* mol, float* out_xyz, int max_atoms, int* out_written);
bool chem_mol_get_render_topology(
        const ChemMol* mol,
        int* out_atomic_numbers,
        int max_atoms,
        int* out_num_atoms,
        int* out_bond_begin,
        int* out_bond_end,
        int* out_bond_order,
        int* out_bond_flags,
        int max_bonds,
        int* out_num_bonds);
bool chem_mol_get_render_coords(const ChemMol* mol, float* out_xyz, int max_atoms, int* out_written);

#ifdef __cplusplus
}
#endif

#endif
