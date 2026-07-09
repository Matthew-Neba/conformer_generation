#include "cffiwrapper.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <vector>

// RDKit and nvMolKit are linked from the micromamba C++ environment.
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/MolPickler.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/Builder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/MolTransforms/MolTransforms.h>
#include <ForceField/ForceField.h>
#include <RDGeneral/RDLog.h>

#include "bfgs_mmff.h"
#include "mmff_properties.h"

using AtomList = std::vector<const RDKit::Atom*>;
using AtomInvariants = std::vector<std::uint32_t>;

struct Torsion {
    unsigned int i;
    unsigned int j;
    unsigned int k;
    unsigned int l;
};

struct TorsionGroup {
    std::vector<Torsion> torsions;
    double max_deviation = 180.0;
};

struct NonRingTorsionBond {
    unsigned int a1;
    unsigned int a2;
    AtomList nb1;
    AtomList nb2;
};

struct ChemMol {
    std::unique_ptr<RDKit::ROMol> mol;
    std::vector<TorsionGroup> nonring_torsions;
    std::vector<TorsionGroup> ring_torsions;
    std::vector<int> atom_idx_to_heavy_idx;
    int conf_id = -1;
};

// get the heavy neighbor atoms for an atom
AtomList heavy_neighbors(const RDKit::Atom* atom, unsigned int exclude_idx) {
    AtomList out;
    for (const auto* neighbor : atom->getOwningMol().atomNeighbors(atom)) {
        if (neighbor->getAtomicNum() > 1 && neighbor->getIdx() != exclude_idx) {
            out.push_back(neighbor);
        }
    }

    return out;
}

// checks if all the atoms in the list are the same
bool atoms_match(const AtomInvariants& invariants, const AtomList& atoms) {
    if (atoms.empty()) {
        return true;
    }

    const std::uint32_t first = invariants[atoms[0]->getIdx()];
    for (size_t i = 1; i < atoms.size(); i++) {
        if (invariants[atoms[i]->getIdx()] != first) {
            return false;
        }
    }

    return true;
}

// checks if no atom is equal to any other atom (all different)
bool atoms_do_not_match(const AtomInvariants& invariants, const AtomList& atoms) {
    for (size_t i = 0; i + 1 < atoms.size(); i++) {
        for (size_t j = i + 1; j < atoms.size(); j++) {
            if (invariants[atoms[i]->getIdx()] == invariants[atoms[j]->getIdx()]) {
                return false;
            }
        }
    }

    return true;
}

// For exactly three neighbors, return the one atom whose invariant differs
// from the other two. Same as RDKit's _doMatchExcept1 helper.
const RDKit::Atom* match_except_one(const AtomInvariants& invariants, const AtomList& atoms) {
    if (atoms.size() != 3) {
        return nullptr;
    }

    const auto a0 = invariants[atoms[0]->getIdx()];
    const auto a1 = invariants[atoms[1]->getIdx()];
    const auto a2 = invariants[atoms[2]->getIdx()];

    if (a0 == a1 && a0 != a2) {
        return atoms[2];
    }
    if (a0 == a2 && a0 != a1) {
        return atoms[1];
    }
    if (a1 == a2 && a1 != a0) {
        return atoms[0];
    }

    return nullptr;
}

void sort_atoms_by_invariant(AtomList& atoms, const AtomInvariants& invariants) {
    std::sort(atoms.begin(), atoms.end(), [&invariants](const RDKit::Atom* a, const RDKit::Atom* b) {
        return invariants[a->getIdx()] < invariants[b->getIdx()];
    });
}

// Pick stable neighbor atom(s) for one side of a torsion.
AtomList index_for_torsion(AtomList neighbors, const AtomInvariants& invariants) {
    if (neighbors.size() <= 1 || atoms_match(invariants, neighbors)) {
        return neighbors;
    }

    if (atoms_do_not_match(invariants, neighbors)) {
        sort_atoms_by_invariant(neighbors, invariants);
        return {neighbors[0]};
    }
    if (neighbors.size() == 3) {
        const RDKit::Atom* atom = match_except_one(invariants, neighbors);
        if (atom != nullptr) {
            return {atom};
        }
    }

    // partially symmetric case: choose a stable reference atom
    // by taking the lowest Morgan invariant as a tiebreaker
    sort_atoms_by_invariant(neighbors, invariants);
    return {neighbors[0]};
}

// Morgan invariants give stable local atom ids for torsion reference selection.
AtomInvariants atom_invariants_with_radius(const RDKit::ROMol& mol, unsigned int radius) {
    AtomInvariants invariants(mol.getNumAtoms(), 0);
    for (const auto* atom : mol.atoms()) {
        // Default for atoms that do not get a Morgan environment at this radius.
        invariants[atom->getIdx()] = (std::uint32_t)atom->getAtomicNum();
    }

    RDKit::MorganFingerprints::BitInfoMap bit_info;
    std::unique_ptr<RDKit::SparseIntVect<std::uint32_t>> fp(RDKit::MorganFingerprints::getFingerprint(
        mol, radius, nullptr, nullptr, false, true, true, false, &bit_info, true));
    // bit_info is the output we care about; fp is kept only so getFingerprint populates it
    (void)fp;

    for (const auto& entry : bit_info) {
        for (const auto& atom_bit_hit : entry.second) {
            // only use Morgan hits at the requested radius, ignore lower-radius hits
            if (atom_bit_hit.second == radius && atom_bit_hit.first < invariants.size()) {
                invariants[atom_bit_hit.first] = entry.first;
            }
        }
    }

    return invariants;
}

bool bond_is_eligible_for_torsion(const RDKit::Bond* bond, const RDKit::RingInfo* ring_info) {
    // Rotating around non-single bonds costs an astronomical amount of energy.
    if (bond->getBondType() != RDKit::Bond::SINGLE) {
        return false;
    }

    // Ring rotatable bonds are excluded from the action space for now.
    // In rings, changing one torsion can cause coupled changes across other ring torsions.
    // Too complex in big rings
    if (ring_info != nullptr && ring_info->numBondRings(bond->getIdx()) > 0) {
        return false;
    }

    return true;
}

// find non-ring central bonds and possible outer atoms for torsions
std::vector<NonRingTorsionBond> nonring_bonds_for_torsions(const RDKit::ROMol& mol) {
    std::vector<NonRingTorsionBond> out;

    const RDKit::RingInfo* ring_info = mol.getRingInfo();
    for (const auto* bond : mol.bonds()) {
        if (!bond_is_eligible_for_torsion(bond, ring_info)) {
            continue;
        }

        const RDKit::Atom* begin = bond->getBeginAtom();
        const RDKit::Atom* end = bond->getEndAtom();
        AtomList nb1 = heavy_neighbors(begin, end->getIdx());
        AtomList nb2 = heavy_neighbors(end, begin->getIdx());
        if (nb1.empty() || nb2.empty()) {
            continue;
        }

        out.push_back({begin->getIdx(), end->getIdx(), nb1, nb2});
    }

    return out;
}

// build all i-j-k-l torsions for a central bond j-k
void add_torsions(TorsionGroup& group, unsigned int a1, unsigned int a2, const AtomList& left, const AtomList& right) {
    for (const RDKit::Atom* left_atom : left) {
        for (const RDKit::Atom* right_atom : right) {
            group.torsions.push_back({left_atom->getIdx(), a1, a2, right_atom->getIdx()});
        }
    }
}

// Group non-ring torsions by central rotatable bond.
std::vector<TorsionGroup> calculate_nonring_torsion_groups(const RDKit::ROMol& mol) {
    std::vector<TorsionGroup> groups;
    const std::vector<NonRingTorsionBond> bonds = nonring_bonds_for_torsions(mol);
    const AtomInvariants invariants = atom_invariants_with_radius(mol, 2);

    for (const NonRingTorsionBond& bond : bonds) {
        const AtomList d1 = index_for_torsion(bond.nb1, invariants);
        const AtomList d2 = index_for_torsion(bond.nb2, invariants);
        TorsionGroup group;
        add_torsions(group, bond.a1, bond.a2, d1, d2);
        if (!group.torsions.empty()) {
            groups.push_back(std::move(group));
        }
    }

    return groups;
}

// Ring torsions are used only for TFD scoring.
std::vector<TorsionGroup> calculate_ring_torsion_groups(const RDKit::ROMol& mol) {
    std::vector<TorsionGroup> groups;
    const RDKit::RingInfo* ring_info = mol.getRingInfo();
    if (ring_info == nullptr) {
        return groups;
    }

    for (const auto& ring : ring_info->atomRings()) {
        const size_t n = ring.size();

        // torsion needs 4 atoms
        if (n < 4) {
            continue;
        }

        TorsionGroup group;
        const double ring_delta = (double)n - 14.0;
        group.max_deviation = n >= 14 ? 180.0 : 180.0 * std::exp(-0.025 * ring_delta * ring_delta);
        for (size_t i = 0; i < n; i++) {
            group.torsions.push_back({
                (unsigned int)ring[i],
                (unsigned int)ring[(i + 1) % n],
                (unsigned int)ring[(i + 2) % n],
                (unsigned int)ring[(i + 3) % n]});
        }
        groups.push_back(std::move(group));
    }

    return groups;
}

// convert a 0-5 action bucket into six 60-degree torsion angles:
// -180, -120, -60, 0, 60, 120. +180 is omitted because it is equivalent to -180.
double bucket_to_degrees(int bucket) {
    return -180.0 + 60.0 * (double)bucket;
}

double circular_degrees_difference(double a, double b) {
    const double diff = std::fabs(a - b);
    return 360.0 - diff < diff ? 360.0 - diff : diff;
}

// Non-ring torsion angles for one torsion group.
std::vector<double> nonring_torsion_angles(const RDKit::Conformer& conformer, const TorsionGroup& group) {
    std::vector<double> angles;
    for (const Torsion& torsion : group.torsions) {
        double angle = MolTransforms::getDihedralDeg(conformer, torsion.i, torsion.j, torsion.k, torsion.l);
        if (angle < 0.0) {
            angle += 360.0;
        }
        angles.push_back(angle);
    }

    return angles;
}

// Ring torsion angle summary for one ring group.
std::vector<double> ring_torsion_angles(const RDKit::Conformer& conformer, const TorsionGroup& group) {
    double angle_sum = 0.0;
    for (const Torsion& torsion : group.torsions) {
        angle_sum += std::fabs(MolTransforms::getDihedralDeg(conformer, torsion.i, torsion.j, torsion.k, torsion.l));
    }

    return {angle_sum / (double)group.torsions.size()};
}

// Choose ring or non-ring angle calculation for TFD.
std::vector<double> tfd_torsion_angles(const RDKit::Conformer& conformer, const TorsionGroup& group, bool ring_group) {
    if (ring_group) {
        return ring_torsion_angles(conformer, group);
    }

    return nonring_torsion_angles(conformer, group);
}

// compare angles on a circle, so 350 and 10 degrees differ by 20
double wrapped_torsion_difference(double a, double b) {
    const double diff = std::fabs(a - b);
    return 360.0 - diff < diff ? 360.0 - diff : diff;
}

// Smallest normalized angle difference for one torsion group.
double torsion_group_deviation(
        const RDKit::Conformer& conf_a,
        const RDKit::Conformer& conf_b,
        const TorsionGroup& group,
        bool ring_group) {
    const std::vector<double> angles_a = tfd_torsion_angles(conf_a, group, ring_group);
    const std::vector<double> angles_b = tfd_torsion_angles(conf_b, group, ring_group);
    double min_diff = 180.0;

    for (double a : angles_a) {
        for (double b : angles_b) {
            const double diff = wrapped_torsion_difference(a, b);
            if (diff < min_diff) {
                min_diff = diff;
            }
        }
    }

    return min_diff / group.max_deviation;
}

// Average normalized min deviations across torsion groups.
double calculate_tfd(
        const RDKit::Conformer& conf_a,
        const RDKit::Conformer& conf_b,
        const std::vector<TorsionGroup>& nonring_groups,
        const std::vector<TorsionGroup>& ring_groups) {
    double deviation_sum = 0.0;
    size_t num_deviations = 0;

    auto accumulate_groups = [&](const std::vector<TorsionGroup>& groups, bool ring_group) {
        for (const TorsionGroup& group : groups) {
            deviation_sum += torsion_group_deviation(conf_a, conf_b, group, ring_group);
            num_deviations++;
        }
    };

    accumulate_groups(nonring_groups, false);
    accumulate_groups(ring_groups, true);

    if (num_deviations == 0) {
        return 0.0;
    }
    return deviation_sum / (double)num_deviations;
}

bool enable_logging(void) {
    static bool configured = false;
    if (!configured) {
        RDLog::InitLogs();
        configured = true;
    }
    return true;
}

std::vector<int> atom_idx_to_heavy_idx_map(const RDKit::ROMol& mol) {
    std::vector<int> out(mol.getNumAtoms(), -1);
    int heavy_idx = 0;
    for (const auto* atom : mol.atoms()) {
        if (atom->getAtomicNum() > 1) {
            out[atom->getIdx()] = heavy_idx;
            heavy_idx++;
        }
    }
    return out;
}

ChemMol* chem_mol_from_prepared_base(std::unique_ptr<RDKit::ROMol> mol);

ChemMol* chem_mol_from_base(std::unique_ptr<RDKit::ROMol> base) {
    enable_logging();
    if (!base) {
        return nullptr;
    }

    std::unique_ptr<RDKit::ROMol> mol(RDKit::MolOps::addHs(*base));
    return chem_mol_from_prepared_base(std::move(mol));
}

ChemMol* chem_mol_from_prepared_base(std::unique_ptr<RDKit::ROMol> mol) {
    enable_logging();
    if (!mol) {
        return nullptr;
    }

    ChemMol* out = new ChemMol;
    out->mol = std::move(mol);
    out->atom_idx_to_heavy_idx = atom_idx_to_heavy_idx_map(*out->mol);
    out->nonring_torsions = calculate_nonring_torsion_groups(*out->mol);
    out->ring_torsions = calculate_ring_torsion_groups(*out->mol);
    if (out->mol->getNumConformers() > 0) {
        out->conf_id = out->mol->getConformer().getId();
    }
    return out;
}

// Basic Molecule constructor from smile string
bool chem_mol_from_smiles(const char* smiles, ChemMol** out_mol) {
    enable_logging();
    *out_mol = nullptr;
    try {
        std::unique_ptr<RDKit::ROMol> base(RDKit::SmilesToMol(smiles));
        ChemMol* mol = chem_mol_from_base(std::move(base));
        if (mol == nullptr) {
            BOOST_LOG(rdWarningLog) << "chem_mol_from_smiles returned null" << std::endl;
            std::abort();
        }
        *out_mol = mol;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_from_smiles failed" << std::endl;
        std::abort();
    }
}

// Basic Molecule constructor from Mol file
bool chem_mol_from_molfile(const char* path, ChemMol** out_mol) {
    enable_logging();
    *out_mol = nullptr;
    try {
        std::unique_ptr<RDKit::ROMol> base(RDKit::MolFileToMol(path, true, false));
        ChemMol* mol = chem_mol_from_base(std::move(base));
        if (mol == nullptr) {
            BOOST_LOG(rdWarningLog) << "chem_mol_from_molfile returned null: path=" << path << std::endl;
            std::abort();
        }
        *out_mol = mol;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_from_molfile failed: path=" << path << std::endl;
        std::abort();
    }
}

bool chem_mol_from_pickle(const char* path, ChemMol** out_mol) {
    enable_logging();
    *out_mol = nullptr;
    try {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return false;
        }

        auto mol = std::make_unique<RDKit::RWMol>();
        RDKit::MolPickler::molFromPickle(in, mol.get());
        ChemMol* out = chem_mol_from_prepared_base(std::move(mol));
        if (out == nullptr || out->conf_id < 0) {
            BOOST_LOG(rdWarningLog) << "chem_mol_from_pickle missing conformer: path="
                                    << path << std::endl;
            std::abort();
        }

        *out_mol = out;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_from_pickle failed: path=" << path << std::endl;
        std::abort();
    }
}

bool chem_mol_to_pickle(const ChemMol* mol, const char* path) {
    enable_logging();
    try {
        if (mol == nullptr || !mol->mol || mol->conf_id < 0) {
            BOOST_LOG(rdWarningLog) << "chem_mol_to_pickle received uninitialized molecule" << std::endl;
            std::abort();
        }

        std::ofstream out(path, std::ios::binary);
        if (!out) {
            BOOST_LOG(rdWarningLog) << "chem_mol_to_pickle failed to open: path="
                                    << path << std::endl;
            std::abort();
        }

        unsigned int pickle_flags =
            (unsigned int)RDKit::PicklerOps::PropertyPickleOptions::AllProps |
            (unsigned int)RDKit::PicklerOps::PropertyPickleOptions::CoordsAsDouble;
        RDKit::MolPickler::pickleMol(
            *mol->mol,
            out,
            pickle_flags);
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_to_pickle failed: path=" << path << std::endl;
        std::abort();
    }
}

bool chem_mol_clone(const ChemMol* src, ChemMol** out_mol) {
    enable_logging();
    *out_mol = nullptr;
    try {
        if (src == nullptr || !src->mol) {
            BOOST_LOG(rdWarningLog) << "chem_mol_clone received null source" << std::endl;
            std::abort();
        }

        ChemMol* out = new ChemMol;
        out->mol = std::make_unique<RDKit::ROMol>(*src->mol);
        out->nonring_torsions = src->nonring_torsions;
        out->ring_torsions = src->ring_torsions;
        out->atom_idx_to_heavy_idx = src->atom_idx_to_heavy_idx;
        out->conf_id = src->conf_id;

        if (out->conf_id >= 0) {
            out->mol->getConformer(out->conf_id);
        }

        *out_mol = out;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_clone failed" << std::endl;
        std::abort();
    }
}

bool chem_mol_clone_conformer(const ChemMol* src, int conf_id, ChemMol** out_mol) {
    enable_logging();
    *out_mol = nullptr;
    try {
        ChemMol* out = new ChemMol;
        out->mol = std::make_unique<RDKit::ROMol>(*src->mol, false, conf_id);
        out->nonring_torsions = src->nonring_torsions;
        out->ring_torsions = src->ring_torsions;
        out->atom_idx_to_heavy_idx = src->atom_idx_to_heavy_idx;
        out->conf_id = conf_id;
        out->mol->getConformer(conf_id);

        *out_mol = out;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_clone_conformer failed: conf_id="
                                << conf_id << std::endl;
        std::abort();
    }
}

// call this exactly once for every ChemMol returned by a constructor
bool chem_mol_free(ChemMol* mol) {
    delete mol;
    return true;
}

bool chem_mol_num_heavy_atoms(const ChemMol* mol, int* out_num_heavy_atoms) {
    try {
        *out_num_heavy_atoms = (int)mol->mol->getNumHeavyAtoms();
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_num_heavy_atoms failed" << std::endl;
        std::abort();
    }
}

bool chem_mol_num_nonring_rotatable_torsions(const ChemMol* mol, int* out_num_torsions) {
    try {
        *out_num_torsions = (int)mol->nonring_torsions.size();
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_num_nonring_rotatable_torsions failed" << std::endl;
        std::abort();
    }
}

bool chem_mol_get_atomic_numbers(const ChemMol* mol, int* out_atomic_numbers, int max_atoms, int* out_written) {
    try {
        int written = 0;
        for (const auto* atom : mol->mol->atoms()) {
            if (atom->getAtomicNum() <= 1) {
                continue;
            }
            if (written >= max_atoms) {
                BOOST_LOG(rdWarningLog) << "chem_mol_get_atomic_numbers exceeded max_atoms="
                                        << max_atoms << std::endl;
                std::abort();
            }

            out_atomic_numbers[written] = atom->getAtomicNum();
            written++;
        }

        *out_written = written;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_get_atomic_numbers failed" << std::endl;
        std::abort();
    }
}

// Embed one initial conformer and remove any prior conformers.
bool chem_mol_init_conformer(ChemMol* mol, unsigned int seed, int* out_conf_id) {
    try {
        RDKit::DGeomHelpers::EmbedParameters params;
        params.randomSeed = (int)seed;
        params.clearConfs = true;
        params.enforceChirality = true;
        params.useExpTorsionAnglePrefs = true;
        params.useBasicKnowledge = true;
        params.useRandomCoords = true;
        params.maxIterations = 1000;

        int conf_id = RDKit::DGeomHelpers::EmbedMolecule(*mol->mol, params);
        if (conf_id < 0) {
            BOOST_LOG(rdWarningLog) << "chem_mol_init_conformer failed to embed: seed="
                                    << seed << std::endl;
            return false;
        }
        mol->conf_id = conf_id;
        *out_conf_id = conf_id;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_init_conformer failed: seed=" << seed << std::endl;
        std::abort();
    }
}

// Archive a copy of the active conformer on the molecule.
bool chem_mol_copy_conformer(ChemMol* mol, int* out_conf_id) {
    try {
        RDKit::Conformer* copy = new RDKit::Conformer(mol->mol->getConformer(mol->conf_id));
        // assignId=true means RDKit owns and deletes this conformer pointer
        int conf_id = (int)mol->mol->addConformer(copy, true);
        if (conf_id < 0) {
            BOOST_LOG(rdWarningLog) << "chem_mol_copy_conformer returned invalid conf_id: source_conf_id="
                                    << mol->conf_id << std::endl;
            std::abort();
        }
        *out_conf_id = conf_id;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_copy_conformer failed: source_conf_id="
                                << mol->conf_id << std::endl;
        std::abort();
    }
}

bool chem_mol_remove_conformer(ChemMol* mol, int conf_id) {
    try {
        mol->mol->removeConformer((unsigned int)conf_id);
        if (mol->conf_id == conf_id) {
            mol->conf_id = -1;
        }
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_remove_conformer failed: conf_id="
                                << conf_id << std::endl;
        std::abort();
    }
}

bool chem_mol_set_active_conformer(ChemMol* mol, int conf_id) {
    try {
        mol->mol->getConformer(conf_id);
        mol->conf_id = conf_id;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_set_active_conformer failed: conf_id="
                                << conf_id << std::endl;
        std::abort();
    }
}

bool chem_mol_get_active_conformer_id(const ChemMol* mol, int* out_conf_id) {
    try {
        if (mol == nullptr || !mol->mol || mol->conf_id < 0) {
            BOOST_LOG(rdWarningLog) << "chem_mol_get_active_conformer_id received uninitialized molecule" << std::endl;
            std::abort();
        }
        mol->mol->getConformer(mol->conf_id);
        *out_conf_id = mol->conf_id;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_get_active_conformer_id failed: conf_id="
                                << (mol == nullptr ? -1 : mol->conf_id) << std::endl;
        std::abort();
    }
}
// Set one non-ring torsion on the active conformer using a 0-5 action bucket.
bool chem_mol_set_torsion_bucket(ChemMol* mol, int torsion_idx, int bucket) {
    try {
        const Torsion& torsion = mol->nonring_torsions.at((size_t)torsion_idx).torsions.at(0);
        RDKit::Conformer& conformer = mol->mol->getConformer(mol->conf_id);
        MolTransforms::setDihedralDeg(
            conformer,
            torsion.i,
            torsion.j,
            torsion.k,
            torsion.l,
            bucket_to_degrees(bucket));
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_set_torsion_bucket failed: torsion_idx="
                                << torsion_idx << " bucket=" << bucket << std::endl;
        std::abort();
    }
}

bool chem_mol_get_torsion_bucket(const ChemMol* mol, int torsion_idx, int* out_bucket) {
    try {
        const Torsion& torsion = mol->nonring_torsions.at((size_t)torsion_idx).torsions.at(0);
        const RDKit::Conformer& conformer = mol->mol->getConformer(mol->conf_id);
        double angle = MolTransforms::getDihedralDeg(conformer, torsion.i, torsion.j, torsion.k, torsion.l);
        if (angle < 0.0) {
            angle += 360.0;
        }

        int best_bucket = 0;
        double best_diff = 1e9;
        for (int bucket = 0; bucket < 6; bucket++) {
            double target = bucket_to_degrees(bucket);
            if (target < 0.0) {
                target += 360.0;
            }
            double diff = circular_degrees_difference(angle, target);
            if (diff < best_diff) {
                best_diff = diff;
                best_bucket = bucket;
            }
        }
        *out_bucket = best_bucket;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_get_torsion_bucket failed: torsion_idx="
                                << torsion_idx << std::endl;
        std::abort();
    }
}

bool chem_mol_minimize_mmff_all_conformers(
        ChemMol* mol,
        int max_iters,
        const int* conf_ids,
        int num_conf_ids,
        double* out_energies) {
    try {
        std::vector<int> mol_conf_ids;
        mol_conf_ids.reserve(mol->mol->getNumConformers());
        for (auto it = mol->mol->beginConformers(); it != mol->mol->endConformers(); ++it) {
            mol_conf_ids.push_back((int)(*it)->getId());
        }

        nvMolKit::MMFFProperties properties;
        properties.variant = "MMFF94s";
        properties.nonBondedThreshold = 10.0;
        properties.ignoreInterfragInteractions = true;

        std::vector<RDKit::ROMol*> mols = {mol->mol.get()};
        nvMolKit::MMFF::MMFFMinimizeResult result = nvMolKit::MMFF::MMFFMinimizeMoleculesConfs(
            mols,
            max_iters,
            1e-4,
            {properties},
            {},
            nvMolKit::BatchHardwareOptions(),
            nvMolKit::BfgsBackend::BATCHED,
            nvMolKit::CoordinateOutput::RDKIT_CONFORMERS);

        if (result.energies.size() != 1 || result.energies[0].size() != mol_conf_ids.size()) {
            BOOST_LOG(rdWarningLog) << "chem_mol_minimize_mmff_all_conformers energy shape mismatch";
            std::abort();
        }

        for (int i = 0; i < num_conf_ids; i++) {
            bool found = false;
            for (size_t conf_idx = 0; conf_idx < mol_conf_ids.size(); conf_idx++) {
                if (mol_conf_ids[conf_idx] == conf_ids[i]) {
                    const double energy = result.energies[0][conf_idx];
                    if (!std::isfinite(energy)) {
                        BOOST_LOG(rdWarningLog) << "chem_mol_minimize_mmff_all_conformers returned non-finite energy";
                        std::abort();
                    }
                    out_energies[i] = energy;
                    found = true;
                    break;
                }
            }
            if (!found) {
                BOOST_LOG(rdWarningLog) << "chem_mol_minimize_mmff_all_conformers missing conf_id="
                                        << conf_ids[i] << std::endl;
                std::abort();
            }
        }
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_minimize_mmff_all_conformers failed: max_iters="
                                << max_iters << std::endl;
        std::abort();
    }
}

bool chem_mol_minimize_mmff_conf(ChemMol* mol, int conf_id, int max_iters) {
    try {
        const auto result = RDKit::MMFF::MMFFOptimizeMolecule(
            *mol->mol, max_iters, "MMFF94s", 10.0, conf_id, true);
        if (result.first < 0) {
            BOOST_LOG(rdWarningLog) << "chem_mol_minimize_mmff_conf failed status=" << result.first
                                    << " conf_id=" << conf_id << std::endl;
            std::abort();
        }
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_minimize_mmff_conf failed: max_iters="
                                << max_iters << " conf_id=" << conf_id << std::endl;
        std::abort();
    }
}

bool chem_mol_energy_mmff(ChemMol* mol, int conf_id, double* out_energy) {
    try {
        RDKit::MMFF::MMFFMolProperties props(*mol->mol, "MMFF94s");
        std::unique_ptr<ForceFields::ForceField> ff(
            RDKit::MMFF::constructForceField(*mol->mol, &props, 10.0, conf_id, true));
        if (!ff) {
            BOOST_LOG(rdWarningLog) << "chem_mol_energy_mmff_for_conf failed: force field unavailable for conf_id="
                                    << conf_id << std::endl;
            std::abort();
        }
        double energy = ff->calcEnergy();
        if (!std::isfinite(energy)) {
            BOOST_LOG(rdWarningLog) << "chem_mol_energy_mmff_for_conf returned non-finite energy: conf_id="
                                    << conf_id << std::endl;
            std::abort();
        }
        *out_energy = energy;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_energy_mmff_for_conf failed: conf_id="
                                << conf_id << std::endl;
        std::abort();
    }
}

// write the four compact heavy-atom ids for one rotatable non-ring torsion
bool chem_mol_get_torsion_atoms(const ChemMol* mol, int torsion_idx, int* out_atoms4) {
    try {
        if (torsion_idx < 0 || (size_t)torsion_idx >= mol->nonring_torsions.size()) {
            BOOST_LOG(rdWarningLog) << "chem_mol_get_torsion_atoms out of range: torsion_idx="
                                    << torsion_idx << std::endl;
            std::abort();
        }
        const Torsion& torsion = mol->nonring_torsions[(size_t)torsion_idx].torsions.at(0);

        const unsigned int atom_ids[4] = {torsion.i, torsion.j, torsion.k, torsion.l};
        for (int i = 0; i < 4; i++) {
            if (atom_ids[i] >= mol->atom_idx_to_heavy_idx.size()) {
                BOOST_LOG(rdWarningLog) << "chem_mol_get_torsion_atoms atom id out of range: atom_id="
                                        << atom_ids[i] << std::endl;
                std::abort();
            }
            const int heavy_idx = mol->atom_idx_to_heavy_idx[atom_ids[i]];
            if (heavy_idx < 0) {
                BOOST_LOG(rdWarningLog) << "chem_mol_get_torsion_atoms atom is not heavy: atom_id="
                                        << atom_ids[i] << std::endl;
                std::abort();
            }
            out_atoms4[i] = heavy_idx;
        }
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_get_torsion_atoms failed: torsion_idx="
                                << torsion_idx << std::endl;
        std::abort();
    }
}

// compute TFD between two stored conformers on the same molecule
bool chem_mol_tfd_between_conformers(const ChemMol* mol, int conf_id_a, int conf_id_b, double* out_tfd) {
    try {
        const RDKit::Conformer& conf_a = mol->mol->getConformer(conf_id_a);
        const RDKit::Conformer& conf_b = mol->mol->getConformer(conf_id_b);
        *out_tfd = calculate_tfd(conf_a, conf_b, mol->nonring_torsions, mol->ring_torsions);
        return true;
    } catch (...) {
        // unable to get the tfd
        BOOST_LOG(rdWarningLog) << "chem_mol_tfd_between_conformers failed: conf_id_a="
                                << conf_id_a << " conf_id_b=" << conf_id_b << std::endl;
        std::abort();
    }
}

// write heavy-atom xyz coordinates from the current conformer into a c arr
bool chem_mol_get_coords(const ChemMol* mol, float* out_xyz, int max_atoms, int* out_written) {
    try {
        const RDKit::Conformer& conformer = mol->mol->getConformer(mol->conf_id);
        int written = 0;

        for (const auto* atom : mol->mol->atoms()) {
            // ignore hydrogens
            if (atom->getAtomicNum() <= 1) {
                continue;
            }
            if (written >= max_atoms) {
                BOOST_LOG(rdWarningLog) << "chem_mol_get_coords exceeded max_atoms="
                                        << max_atoms << " conf_id=" << mol->conf_id << std::endl;
                std::abort();
            }

            const RDGeom::Point3D& pos = conformer.getAtomPos(atom->getIdx());
            out_xyz[3 * written + 0] = (float)pos.x;
            out_xyz[3 * written + 1] = (float)pos.y;
            out_xyz[3 * written + 2] = (float)pos.z;
            written++;
        }

        *out_written = written;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_get_coords failed: conf_id=" << mol->conf_id << std::endl;
        std::abort();
    }
}

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
        int* out_num_bonds) {
    try {
        int num_atoms = (int)mol->mol->getNumAtoms();
        if (num_atoms > max_atoms) {
            BOOST_LOG(rdWarningLog) << "chem_mol_get_render_topology exceeded max_atoms="
                                    << max_atoms << " num_atoms=" << num_atoms << std::endl;
            std::abort();
        }

        for (const auto* atom : mol->mol->atoms()) {
            out_atomic_numbers[atom->getIdx()] = atom->getAtomicNum();
        }
        *out_num_atoms = num_atoms;

        int written_bonds = 0;
        for (const auto* bond : mol->mol->bonds()) {
            if (written_bonds >= max_bonds) {
                BOOST_LOG(rdWarningLog) << "chem_mol_get_render_topology exceeded max_bonds="
                                        << max_bonds << std::endl;
                std::abort();
            }

            int order = 1;
            switch (bond->getBondType()) {
                case RDKit::Bond::DOUBLE:
                    order = 2;
                    break;
                case RDKit::Bond::TRIPLE:
                    order = 3;
                    break;
                default:
                    order = 1;
                    break;
            }

            int flags = 0;
            if (bond->getIsAromatic()) {
                flags |= 1;
            }
            if (mol->mol->getRingInfo()->numBondRings(bond->getIdx()) > 0) {
                flags |= 2;
            }

            out_bond_begin[written_bonds] = (int)bond->getBeginAtomIdx();
            out_bond_end[written_bonds] = (int)bond->getEndAtomIdx();
            out_bond_order[written_bonds] = order;
            out_bond_flags[written_bonds] = flags;
            written_bonds++;
        }

        *out_num_bonds = written_bonds;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_get_render_topology failed" << std::endl;
        std::abort();
    }
}

bool chem_mol_get_render_coords(const ChemMol* mol, float* out_xyz, int max_atoms, int* out_written) {
    try {
        const RDKit::Conformer& conformer = mol->mol->getConformer(mol->conf_id);
        int num_atoms = (int)mol->mol->getNumAtoms();
        if (num_atoms > max_atoms) {
            BOOST_LOG(rdWarningLog) << "chem_mol_get_render_coords exceeded max_atoms="
                                    << max_atoms << " num_atoms=" << num_atoms
                                    << " conf_id=" << mol->conf_id << std::endl;
            std::abort();
        }

        for (const auto* atom : mol->mol->atoms()) {
            const RDGeom::Point3D& pos = conformer.getAtomPos(atom->getIdx());
            out_xyz[3 * atom->getIdx() + 0] = (float)pos.x;
            out_xyz[3 * atom->getIdx() + 1] = (float)pos.y;
            out_xyz[3 * atom->getIdx() + 2] = (float)pos.z;
        }

        *out_written = num_atoms;
        return true;
    } catch (...) {
        BOOST_LOG(rdWarningLog) << "chem_mol_get_render_coords failed: conf_id="
                                << mol->conf_id << std::endl;
        std::abort();
    }
}
