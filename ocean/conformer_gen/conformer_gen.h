#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "raylib.h"
#include "rdkit/cffiwrapper.h"

// rendering
#define MAX_RENDER_ATOMS 512
#define MAX_RENDER_BONDS 512
#ifndef SKIP_TERMINAL_MMFF_REWARD
#define SKIP_TERMINAL_MMFF_REWARD 0
#endif

#define RENDER_TARGET_FPS 120
#define RENDER_BOND_AROMATIC 1


// dataset and persistent RDKit template cache
#define CONFORMER_DATASET_ROOT "ocean/conformer_gen/dataset"
#define CONFORMER_CACHE_DIR "ocean/conformer_gen/cache"
#define CONFORMER_RDMOL_CACHE_DIR "ocean/conformer_gen/cache/rdmol_v1"

// molecule family selector from config/conformer_gen.ini
#define MOLECULE_FAMILY_ALKANE 0
#define MOLECULE_FAMILY_LIGNIN 1
#define MOLECULE_FAMILY_BOTH 2

// fixed env bounds for padded arrays and episode limits
#define MAX_HEAVY_ATOMS 130
#define MAX_NONRING_ROTATABLE_TORSIONS 65
#define MAX_PROPOSALS 80

// coordinate normalization
#define COORD_SCALE 10.0f
#define MAX_ATOMIC_NUMBER 118.0f

// reward hypers
#define MMFF_MAX_ITERS 500
#define DEFAULT_BASE_PROPOSAL_REWARD 0.01f
#define DEFAULT_OUT_OF_BOUNDS_EDIT_PENALTY 0.001f
#define DUPLICATE_PROPOSE_PENALTY 0.001f

// actions: set one non-ring torsion to one of six 60-degree buckets, or propose current conformer
#define ACTIONS_PER_TORSION 6
#define NUM_TORSION_ACTIONS (ACTIONS_PER_TORSION * MAX_NONRING_ROTATABLE_TORSIONS)
#define ACTION_PROPOSE NUM_TORSION_ACTIONS
#define NUM_ACTIONS (ACTION_PROPOSE + 1)

// Give the agent enough edit actions to touch every torsion once before each proposal.
#define MAX_STEPS ((MAX_NONRING_ROTATABLE_TORSIONS + 1) * MAX_PROPOSALS)

// observations:
// globals: heavy atom count, torsion count, proposals, accepts
// atoms: valid mask, atomic number, x/y/z for each padded heavy atom slot
// torsions: valid mask, gathered i/j/k/l atom features, sin/cos current bucket
#define GLOBAL_OBS_SIZE 4
#define ATOM_OBS_SIZE 5
#define TORSION_ATOM_FEATURE_SIZE 4
#define TORSION_OBS_SIZE (1 + 4 * TORSION_ATOM_FEATURE_SIZE + 2)
#define ATOM_OBS_OFFSET GLOBAL_OBS_SIZE
#define TORSION_OBS_OFFSET (ATOM_OBS_OFFSET + MAX_HEAVY_ATOMS * ATOM_OBS_SIZE)
#define OBS_SIZE (TORSION_OBS_OFFSET + MAX_NONRING_ROTATABLE_TORSIONS * TORSION_OBS_SIZE)

#define PI_F 3.14159265358979323846f

typedef struct {
    Camera3D camera;
    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;

    int topology_molecule_index;
    int num_atoms;
    int num_bonds;
    int atomic_numbers[MAX_RENDER_ATOMS];
    int bond_begin[MAX_RENDER_BONDS];
    int bond_end[MAX_RENDER_BONDS];
    int bond_order[MAX_RENDER_BONDS];
    int bond_flags[MAX_RENDER_BONDS];
    float coords[MAX_RENDER_ATOMS][3];
    float molecule_radius;

    int seen_episode_length;
    int seen_proposed;
    int seen_accepted;
    int last_status;
    double last_mmff_energy;
    double last_gibbs;
    double last_log_gibbs;
    double proposed_mmff_sum;
    double proposed_gibbs_sum;
    double proposed_log_gibbs_sum;
    double accepted_mmff_sum;
    double accepted_gibbs_sum;
    double accepted_log_gibbs_sum;
} Client;

typedef struct {
    float perf;
    float score;
    float proposed;
    float accepted;
    float episode_return;
    float episode_length;
    float mmff_reward;
    float mmff_reward_ms;
    float duplicate_check_ms;
    float set_torsion_ms;
    float coords_ms;
    float obs_ms;
    float copy_conf_ms;
    float sync_torsion_ms;
    float load_selected_ms;
    float propose_ms;
    float apply_action_ms;
    float step_ms;
    float accounted_ms;
    float reset_init_ms;
    float n;
} Log;

typedef struct {
    char* json_path;
    ChemMol* template_mol;
    int template_conf_id;
    double e0;
    double z0;
    double tau;
    int n_heavy_atoms;
    int n_nonring_rotatable_torsions;
    int atomic_numbers[MAX_HEAVY_ATOMS];
    int nonring_rotatable_torsion_atoms[MAX_NONRING_ROTATABLE_TORSIONS][4];
} ConformerMolecule;

typedef struct {
    Log log;

    float* observations;
    float* actions;
    float* rewards;
    float* terminals;

    int num_agents;
    unsigned int rng;
    int owns_buffers;

    int episode_length;
    float episode_return;

    int num_proposed;
    int num_accepted;
    int pending_reset;
    int molecule_index;
    int molecule_family;
    int num_molecules;
    int molecule_capacity;
    ConformerMolecule* molecules;
    Client* client;
    ChemMol* mol_handle;
    int working_conf_id;
    double e0;
    double z0;
    double tau;
    int n_heavy_atoms;
    int n_nonring_rotatable_torsions;
    int atomic_numbers[MAX_HEAVY_ATOMS];
    int nonring_rotatable_torsion_atoms[MAX_NONRING_ROTATABLE_TORSIONS][4];

    // written to observations
    int current_torsion_bucket[MAX_NONRING_ROTATABLE_TORSIONS];
    float working_coords[MAX_HEAVY_ATOMS][3];

    int accepted_conf_ids[MAX_PROPOSALS];
    uint8_t accepted_torsion_buckets[MAX_PROPOSALS][MAX_NONRING_ROTATABLE_TORSIONS];
    double episode_mmff_reward;
    double episode_mmff_reward_ms;
    double episode_duplicate_check_ms;
    double episode_set_torsion_ms;
    double episode_coords_ms;
    double episode_obs_ms;
    double episode_copy_conf_ms;
    double episode_sync_torsion_ms;
    double episode_load_selected_ms;
    double episode_propose_ms;
    double episode_apply_action_ms;
    double episode_step_ms;
    double episode_reset_init_ms;
} ConformerGen;

// Return a monotonic timestamp in milliseconds for profiling work.
static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static float clamp_float(float value, float lo, float hi) {
    return fminf(fmaxf(value, lo), hi);
}

// Check whether a string ends with the given suffix.
static bool has_suffix(const char* s, const char* suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

// Check whether a filesystem path points to a directory.
static bool path_is_directory(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

// Allocate and return a joined filesystem path from two path parts.
static char* path_join(const char* a, const char* b) {
    if (b[0] == '/') {
        char* out = (char*)malloc(strlen(b) + 1);
        strcpy(out, b);
        return out;
    }

    size_t na = strlen(a);
    size_t nb = strlen(b);
    int need_slash = na > 0 && a[na - 1] != '/';
    char* out = (char*)malloc(na + (size_t)need_slash + nb + 1);

    memcpy(out, a, na);
    if (need_slash) {
        out[na] = '/';
    }
    memcpy(out + na + (size_t)need_slash, b, nb + 1);
    return out;
}

// Read a whole text file into a newly allocated null-terminated buffer.
static char* read_text_file(const char* path) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    if (size < 0) {
        fclose(file);
        return NULL;
    }
    rewind(file);

    char* data = (char*)malloc((size_t)size + 1);
    if (data == NULL) {
        fclose(file);
        return NULL;
    }
    size_t read = fread(data, 1, (size_t)size, file);
    fclose(file);
    if (read != (size_t)size) {
        free(data);
        return NULL;
    }
    data[size] = '\0';
    return data;
}

// Find the start of a JSON value for a simple top-level key.
static const char* json_find_key(const char* json, const char* key) {
    char pattern[128];
    int n = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* pos = strstr(json, pattern);
    if (pos == NULL) {
        return NULL;
    }

    pos += n;
    while (*pos != '\0' && isspace((unsigned char)*pos)) {
        pos++;
    }
    if (*pos != ':') {
        return NULL;
    }
    pos++;
    while (*pos != '\0' && isspace((unsigned char)*pos)) {
        pos++;
    }
    return pos;
}

// Parse an optional string value from a simple JSON object.
static char* json_string_optional(const char* json, const char* key) {
    const char* pos = json_find_key(json, key);
    if (pos == NULL || *pos != '"') {
        return NULL;
    }
    pos++;

    const char* end = pos;
    while (*end != '\0' && *end != '"') {
        end++;
    }
    if (*end != '"') {
        return NULL;
    }

    size_t n = (size_t)(end - pos);
    char* out = (char*)malloc(n + 1);
    memcpy(out, pos, n);
    out[n] = '\0';
    return out;
}

static bool json_double_required(const char* json, const char* key, double* out) {
    const char* pos = json_find_key(json, key);
    if (pos == NULL) {
        return false;
    }

    char* end = NULL;
    double value = strtod(pos, &end);
    if (end == pos) {
        return false;
    }

    *out = value;
    return true;
}

static void add_molecule(ConformerGen* env, ConformerMolecule molecule) {
    if (env->num_molecules >= env->molecule_capacity) {
        int new_capacity = env->molecule_capacity == 0 ? 64 : env->molecule_capacity * 2;
        ConformerMolecule* molecules = (ConformerMolecule*)realloc(
            env->molecules, (size_t)new_capacity * sizeof(ConformerMolecule));
        env->molecules = molecules;
        env->molecule_capacity = new_capacity;
    }

    env->molecules[env->num_molecules] = molecule;
    env->num_molecules++;
}

// Check whether a dataset path belongs to the selected molecule family.
static bool family_matches_path(int molecule_family, const char* path) {
    if (molecule_family == MOLECULE_FAMILY_BOTH) {
        return true;
    }
    if (molecule_family == MOLECULE_FAMILY_LIGNIN) {
        return strstr(path, "lignin") != NULL;
    }
    return molecule_family == MOLECULE_FAMILY_ALKANE && strstr(path, "alkane") != NULL;
}

// Check whether a dataset path looks like a training split.
static bool is_train_path(const char* path) {
    return strstr(path, "_train") != NULL || strstr(path, "/train") != NULL;
}

static unsigned int path_seed(const char* path) {
    uint32_t hash = 2166136261u;
    for (const unsigned char* p = (const unsigned char*)path; *p != '\0'; p++) {
        hash ^= (uint32_t)(*p);
        hash *= 16777619u;
    }
    hash &= 0x7fffffffU;
    return hash == 0 ? 1U : hash;
}

static void ensure_directory(const char* path) {
    mkdir(path, 0775);
}

static void ensure_molecule_cache_dir(void) {
    ensure_directory(CONFORMER_CACHE_DIR);
    ensure_directory(CONFORMER_RDMOL_CACHE_DIR);
}

static char* molecule_cache_path(const char* json_path) {
    char filename[64];
    snprintf(filename, sizeof(filename), "%08x.rdmol", path_seed(json_path));
    return path_join(CONFORMER_RDMOL_CACHE_DIR, filename);
}

static bool file_mtime(const char* path, time_t* out_mtime) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return false;
    }
    *out_mtime = st.st_mtime;
    return true;
}

static bool cache_is_fresh(const char* cache_path, const char* json_path, const char* molfile_path) {
    time_t cache_mtime = 0;
    time_t source_mtime = 0;
    if (!file_mtime(cache_path, &cache_mtime) || !file_mtime(json_path, &source_mtime)) {
        return false;
    }
    if (cache_mtime < source_mtime) {
        return false;
    }
    if (molfile_path != NULL) {
        if (!file_mtime(molfile_path, &source_mtime) || cache_mtime < source_mtime) {
            return false;
        }
    }
    return true;
}

static void free_molecule_template(ConformerMolecule* molecule) {
    if (molecule->template_mol != NULL) {
        chem_mol_free(molecule->template_mol);
    }
    molecule->template_mol = NULL;
    free(molecule->json_path);
    molecule->json_path = NULL;
}

static bool finish_molecule_template(ConformerMolecule* molecule) {
    if (molecule->template_mol == NULL) {
        return false;
    }

    chem_mol_num_heavy_atoms(molecule->template_mol, &molecule->n_heavy_atoms);
    chem_mol_num_nonring_rotatable_torsions(
        molecule->template_mol, &molecule->n_nonring_rotatable_torsions);
    if (molecule->n_heavy_atoms <= 0 || molecule->n_heavy_atoms > MAX_HEAVY_ATOMS ||
            molecule->n_nonring_rotatable_torsions < 0 ||
            molecule->n_nonring_rotatable_torsions > MAX_NONRING_ROTATABLE_TORSIONS) {
        return false;
    }

    int written_atomic_numbers = 0;
    chem_mol_get_atomic_numbers(
        molecule->template_mol, molecule->atomic_numbers, MAX_HEAVY_ATOMS, &written_atomic_numbers);
    if (written_atomic_numbers != molecule->n_heavy_atoms) {
        return false;
    }

    for (int i = 0; i < molecule->n_nonring_rotatable_torsions; i++) {
        chem_mol_get_torsion_atoms(
            molecule->template_mol, i, molecule->nonring_rotatable_torsion_atoms[i]);
    }

    return true;
}

static bool preprocess_molecule_template(
        ConformerMolecule* molecule, const char* smiles, const char* molfile_path, const char* cache_path) {
    if (cache_is_fresh(cache_path, molecule->json_path, molfile_path) &&
            chem_mol_from_pickle(cache_path, &molecule->template_mol)) {
        chem_mol_get_active_conformer_id(molecule->template_mol, &molecule->template_conf_id);
        return finish_molecule_template(molecule);
    }

    if (molfile_path != NULL) {
        chem_mol_from_molfile(molfile_path, &molecule->template_mol);
    } else {
        chem_mol_from_smiles(smiles, &molecule->template_mol);
    }
    if (molecule->template_mol == NULL) {
        return false;
    }

    molecule->template_conf_id = -1;
    if (!chem_mol_init_conformer(
            molecule->template_mol, path_seed(molecule->json_path), &molecule->template_conf_id)) {
        return false;
    }

    if (!finish_molecule_template(molecule)) {
        return false;
    }

    ensure_molecule_cache_dir();
    chem_mol_to_pickle(molecule->template_mol, cache_path);
    return true;
}

// Load one molecule metadata JSON file and append it to the environment.
static bool load_molecule_json(ConformerGen* env, const char* dir_path, const char* filename) {
    char* json_path = path_join(dir_path, filename);
    ConformerMolecule molecule;
    memset(&molecule, 0, sizeof(molecule));
    molecule.json_path = json_path;

    char* json = read_text_file(json_path);
    char* smiles = NULL;
    char* molfile_path = NULL;
    char* cache_path = molecule_cache_path(json_path);
    if (json == NULL) {
        free(cache_path);
        free_molecule_template(&molecule);
        return false;
    }

    char* molfile = json_string_optional(json, "molfile");
    if (molfile != NULL) {
        molfile_path = path_join(dir_path, molfile);
        free(molfile);
    } else {
        smiles = json_string_optional(json, "smiles");
    }

    bool loaded =
        (molfile_path != NULL || smiles != NULL) &&
        json_double_required(json, "E0", &molecule.e0) &&
        json_double_required(json, "Z0", &molecule.z0) &&
        json_double_required(json, "tau", &molecule.tau) &&
        molecule.z0 > 0.0 &&
        molecule.tau > 0.0 &&
        preprocess_molecule_template(&molecule, smiles, molfile_path, cache_path);

    free(json);
    free(smiles);
    free(molfile_path);
    free(cache_path);
    if (loaded) {
        add_molecule(env, molecule);
        return true;
    }
    free_molecule_template(&molecule);
    return false;
}

// Recursively scan the dataset tree and load matching training molecule JSON files.
static void scan_dataset_dir(ConformerGen* env, const char* dir_path) {
    DIR* dir = opendir(dir_path);
    if (dir == NULL) {
        return;
    }

    struct dirent* entry = NULL;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        char* path = path_join(dir_path, entry->d_name);
        if (path_is_directory(path)) {
            scan_dataset_dir(env, path);
        } else if (is_train_path(dir_path) &&
                family_matches_path(env->molecule_family, dir_path) &&
                has_suffix(entry->d_name, ".json") &&
                strcmp(entry->d_name, "summary.json") != 0) {
            load_molecule_json(env, dir_path, entry->d_name);
        }
        free(path);
    }

    closedir(dir);
}

// Lazily load and cache conformer molecules for the selected molecule family. (shared for all vecenvs)
static void load_conformer_molecules(ConformerGen* env) {
    static ConformerMolecule* molecules[3] = {NULL, NULL, NULL};
    static int num_molecules[3] = {0, 0, 0};

    int family = env->molecule_family;
    if (molecules[family] != NULL) {
        env->molecules = molecules[family];
        env->num_molecules = num_molecules[family];
        env->molecule_capacity = num_molecules[family];
        return;
    }

    scan_dataset_dir(env, CONFORMER_DATASET_ROOT);
    if (env->num_molecules == 0) {
        fprintf(stderr, "conformer_gen: no conformer train molecules loaded\n");
        abort();
    }

    molecules[family] = env->molecules;
    num_molecules[family] = env->num_molecules;
}

// Convert a torsion bucket index into its target angle in radians.
static float torsion_bucket_radians(int bucket) {
    float degrees = -180.0f + 60.0f * (float)bucket;
    return degrees * PI_F / 180.0f;
}

// Write one atom's atomic number and coordinates into a torsion observation slot, (leave at 0 if our current molecule does not have the atom)
static void write_atom_feature(float* observations, int offset, ConformerGen* env, int atom_idx) {
    if (atom_idx < 0 || atom_idx >= env->n_heavy_atoms) {
        return;
    }
    
    observations[offset + 0] = (float)env->atomic_numbers[atom_idx] / MAX_ATOMIC_NUMBER;
    observations[offset + 1] = env->working_coords[atom_idx][0];
    observations[offset + 2] = env->working_coords[atom_idx][1];
    observations[offset + 3] = env->working_coords[atom_idx][2];
}

// Scale molecule coordinates in place.
static void normalize_coords(float coords[MAX_HEAVY_ATOMS][3], int n) {
    for (int i = 0; i < n; i++) {
        coords[i][0] = coords[i][0] / COORD_SCALE;
        coords[i][1] = coords[i][1] / COORD_SCALE;
        coords[i][2] = coords[i][2] / COORD_SCALE;
    }
}

// Refresh normalized working coordinates from the current RDKit conformer.
static void update_working_coords(ConformerGen* env) {
    double coords_start_ms = monotonic_ms();
    memset(env->working_coords, 0, sizeof(env->working_coords));
    int written = 0;
    chem_mol_get_coords(env->mol_handle, &env->working_coords[0][0], MAX_HEAVY_ATOMS, &written);
    if (written != env->n_heavy_atoms) {
        fprintf(stderr, "conformer_gen: coordinate count mismatch: expected=%d written=%d\n",
            env->n_heavy_atoms, written);
        abort();
    }

    normalize_coords(env->working_coords, env->n_heavy_atoms);
    env->episode_coords_ms += monotonic_ms() - coords_start_ms;
}

// Read current torsion buckets from RDKit into environment state.
static void sync_torsion_buckets_from_conformer(ConformerGen* env) {
    double sync_torsion_start_ms = monotonic_ms();
    for (int i = 0; i < env->n_nonring_rotatable_torsions; i++) {
        int bucket = -1;
        chem_mol_get_torsion_bucket(env->mol_handle, i, &bucket);
        if (bucket < 0 || bucket >= ACTIONS_PER_TORSION) {
            fprintf(stderr, "conformer_gen: invalid torsion bucket: torsion=%d bucket=%d\n", i, bucket);
            abort();
        }
        env->current_torsion_bucket[i] = bucket;
    }
    env->episode_sync_torsion_ms += monotonic_ms() - sync_torsion_start_ms;
}

// Fill the observation buffer from the current molecule and episode state.
static void compute_observations(ConformerGen* env) {
    double obs_start_ms = monotonic_ms();

    // zero out the observations
    memset(env->observations, 0, (size_t)OBS_SIZE * sizeof(*env->observations));

    env->observations[0] = (float)env->n_heavy_atoms / (float)MAX_HEAVY_ATOMS;
    env->observations[1] = (float)env->n_nonring_rotatable_torsions / (float)MAX_NONRING_ROTATABLE_TORSIONS;
    env->observations[2] = (float)env->num_proposed / (float)MAX_PROPOSALS;
    env->observations[3] = (float)env->num_accepted / (float)MAX_PROPOSALS;

    for (int atom_idx = 0; atom_idx < env->n_heavy_atoms; atom_idx++) {
        int offset = ATOM_OBS_OFFSET + atom_idx * ATOM_OBS_SIZE;
        env->observations[offset + 0] = 1.0f;
        env->observations[offset + 1] = (float)env->atomic_numbers[atom_idx] / MAX_ATOMIC_NUMBER;
        env->observations[offset + 2] = env->working_coords[atom_idx][0];
        env->observations[offset + 3] = env->working_coords[atom_idx][1];
        env->observations[offset + 4] = env->working_coords[atom_idx][2];
    }

    for (int torsion_idx = 0; torsion_idx < env->n_nonring_rotatable_torsions; torsion_idx++) {
        int offset = TORSION_OBS_OFFSET + torsion_idx * TORSION_OBS_SIZE;
        env->observations[offset + 0] = 1.0f;

        int feature_offset = offset + 1;
        for (int atom_slot = 0; atom_slot < 4; atom_slot++) {
            int atom_idx = env->nonring_rotatable_torsion_atoms[torsion_idx][atom_slot];
            write_atom_feature(env->observations, feature_offset, env, atom_idx);
            feature_offset += TORSION_ATOM_FEATURE_SIZE;
        }

        float radians = torsion_bucket_radians(env->current_torsion_bucket[torsion_idx]);
        env->observations[offset + TORSION_OBS_SIZE - 2] = sinf(radians);
        env->observations[offset + TORSION_OBS_SIZE - 1] = cosf(radians);
    }

    env->episode_obs_ms += monotonic_ms() - obs_start_ms;
}

// Allocate standalone buffers and load shared molecule metadata.
void allocate(ConformerGen* env) {
    env->num_agents = 1;
    env->client = NULL;
    env->mol_handle = NULL;
    env->molecule_index = -1;
    env->working_conf_id = -1;
    env->pending_reset = 0;
    load_conformer_molecules(env);

    env->observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (float*)calloc(1, sizeof(float));
    env->owns_buffers = 1;
}

// Free the current RDKit molecule and any buffers owned by this environment.
void deallocate(ConformerGen* env) {
    if (env->mol_handle != NULL) {
        chem_mol_free(env->mol_handle);
    }

    if (env->owns_buffers) {
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
    }

    env->mol_handle = NULL;
    env->working_conf_id = -1;
    env->molecule_index = -1;
    env->observations = NULL;
    env->actions = NULL;
    env->rewards = NULL;
    env->terminals = NULL;
    env->owns_buffers = 0;
}

// Accumulate one finished episode's metrics into the log buffer.
void add_log(ConformerGen* env) {
    float episode_perf = env->num_proposed > 0
        ? env->episode_return / (float)env->num_proposed
        : 0.0f;

    // score: total episode reward. perf: reward per proposal.
    env->log.perf += episode_perf;
    env->log.score += env->episode_return;
    env->log.proposed += (float)env->num_proposed;
    env->log.accepted += (float)env->num_accepted;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->episode_length;
    env->log.mmff_reward += (float)env->episode_mmff_reward;
    env->log.mmff_reward_ms += (float)env->episode_mmff_reward_ms;
    env->log.duplicate_check_ms += (float)env->episode_duplicate_check_ms;
    env->log.set_torsion_ms += (float)env->episode_set_torsion_ms;
    env->log.coords_ms += (float)env->episode_coords_ms;
    env->log.obs_ms += (float)env->episode_obs_ms;
    env->log.copy_conf_ms += (float)env->episode_copy_conf_ms;
    env->log.sync_torsion_ms += (float)env->episode_sync_torsion_ms;
    env->log.load_selected_ms += (float)env->episode_load_selected_ms;
    env->log.propose_ms += (float)env->episode_propose_ms;
    env->log.apply_action_ms += (float)env->episode_apply_action_ms;
    env->log.step_ms += (float)env->episode_step_ms;
    env->log.accounted_ms += (float)(
        env->episode_step_ms +
        env->episode_reset_init_ms);
    env->log.reset_init_ms += (float)env->episode_reset_init_ms;
    env->log.n += 1.0f;
}

// Clone one preinitialized RDKit molecule template into this mutable env instance.
static void load_selected_molecule(ConformerGen* env, const ConformerMolecule* molecule) {
    double load_selected_start_ms = monotonic_ms();
    if (env->mol_handle != NULL) {
        chem_mol_free(env->mol_handle);
        env->mol_handle = NULL;
    }

    chem_mol_clone(molecule->template_mol, &env->mol_handle);

    env->working_conf_id = molecule->template_conf_id;
    env->e0 = molecule->e0;
    env->z0 = molecule->z0;
    env->tau = molecule->tau;
    env->n_heavy_atoms = molecule->n_heavy_atoms;
    env->n_nonring_rotatable_torsions = molecule->n_nonring_rotatable_torsions;
    memcpy(env->atomic_numbers, molecule->atomic_numbers, sizeof(env->atomic_numbers));
    memcpy(env->nonring_rotatable_torsion_atoms,
        molecule->nonring_rotatable_torsion_atoms,
        sizeof(env->nonring_rotatable_torsion_atoms));

    chem_mol_set_active_conformer(env->mol_handle, env->working_conf_id);
    update_working_coords(env);
    sync_torsion_buckets_from_conformer(env);
    env->episode_load_selected_ms += monotonic_ms() - load_selected_start_ms;
}

static int choose_reset_molecule_index(ConformerGen* env) {
    if (env->num_molecules <= 1) {
        return 0;
    }

    int previous = env->molecule_index;
    if (previous < 0 || previous >= env->num_molecules) {
        return (int)(rand_r(&env->rng) % (unsigned int)env->num_molecules);
    }

    unsigned int span = (unsigned int)(env->num_molecules - 1);
    int draw = (int)(rand_r(&env->rng) % span);
    return draw >= previous ? draw + 1 : draw;
}

static void initialize_base_molecule(ConformerGen* env) {
    int molecule_index = choose_reset_molecule_index(env);
    load_selected_molecule(env, &env->molecules[molecule_index]);
    env->molecule_index = molecule_index;
}

// Reset episode counters and restore a fresh clone of the cached initial conformer.
void c_reset(ConformerGen* env) {
    if (env->observations == NULL) {
        allocate(env);
    }

    double reset_init_start_ms = monotonic_ms();

    env->episode_length = 0;
    env->episode_return = 0.0f;
    env->num_proposed = 0;
    env->num_accepted = 0;
    env->pending_reset = 0;
    env->episode_mmff_reward = 0.0;
    env->episode_mmff_reward_ms = 0.0;
    env->episode_duplicate_check_ms = 0.0;
    env->episode_set_torsion_ms = 0.0;
    env->episode_coords_ms = 0.0;
    env->episode_obs_ms = 0.0;
    env->episode_copy_conf_ms = 0.0;
    env->episode_sync_torsion_ms = 0.0;
    env->episode_load_selected_ms = 0.0;
    env->episode_propose_ms = 0.0;
    env->episode_apply_action_ms = 0.0;
    env->episode_step_ms = 0.0;
    env->episode_reset_init_ms = 0.0;

    memset(env->accepted_conf_ids, 0, sizeof(env->accepted_conf_ids));
    memset(env->accepted_torsion_buckets, 0, sizeof(env->accepted_torsion_buckets));
    initialize_base_molecule(env);
    compute_observations(env);
    env->episode_reset_init_ms += monotonic_ms() - reset_init_start_ms;
}

// Archive the current conformation proposal if its torsion bucket state is new.
static float propose_conformer(ConformerGen* env) {
    double propose_start_ms = monotonic_ms();
    double duplicate_check_start_ms = monotonic_ms();

    bool duplicate = false;
    for (int i = 0; i < env->num_accepted; i++) {
        bool matches = true;
        for (int j = 0; j < env->n_nonring_rotatable_torsions; j++) {
            if ((uint8_t)env->current_torsion_bucket[j] != env->accepted_torsion_buckets[i][j]) {
                matches = false;
                break;
            }
        }
        if (matches) {
            duplicate = true;
            break;
        }
    }
    env->episode_duplicate_check_ms += monotonic_ms() - duplicate_check_start_ms;

    env->num_proposed += 1;
    if (duplicate) {
        env->episode_propose_ms += monotonic_ms() - propose_start_ms;
        return DUPLICATE_PROPOSE_PENALTY;
    }
    int conf_id = -1;
    double copy_conf_start_ms = monotonic_ms();
    chem_mol_copy_conformer(env->mol_handle, &conf_id);
    env->episode_copy_conf_ms += monotonic_ms() - copy_conf_start_ms;
    env->accepted_conf_ids[env->num_accepted] = conf_id;
    for (int i = 0; i < env->n_nonring_rotatable_torsions; i++) {
        env->accepted_torsion_buckets[env->num_accepted][i] = (uint8_t)env->current_torsion_bucket[i];
    }
    env->num_accepted += 1;

    env->episode_propose_ms += monotonic_ms() - propose_start_ms;

    // this is the base reward we get for doing a proposalS
    return DEFAULT_BASE_PROPOSAL_REWARD;
}

// Apply the agent action as either a torsion edit or conformer proposal.
static float apply_action(ConformerGen* env) {
    double apply_action_start_ms = monotonic_ms();
    int action = (int)env->actions[0];
    if (action == ACTION_PROPOSE) {
        float reward = propose_conformer(env);
        env->episode_apply_action_ms += monotonic_ms() - apply_action_start_ms;
        return reward;
    }

    // The action space is padded to MAX_NONRING_ROTATABLE_TORSIONS.
    int torsion_idx = action / ACTIONS_PER_TORSION;
    int bucket = action % ACTIONS_PER_TORSION;
    if (torsion_idx >= env->n_nonring_rotatable_torsions) {
        env->episode_apply_action_ms += monotonic_ms() - apply_action_start_ms;
        return DEFAULT_OUT_OF_BOUNDS_EDIT_PENALTY;
    }

    double set_torsion_start_ms = monotonic_ms();
    chem_mol_set_torsion_bucket(env->mol_handle, torsion_idx, bucket);
    env->episode_set_torsion_ms += monotonic_ms() - set_torsion_start_ms;
    env->current_torsion_bucket[torsion_idx] = bucket;
    update_working_coords(env);

    env->episode_apply_action_ms += monotonic_ms() - apply_action_start_ms;

    return 0.0f;
}

static float terminal_mmff_reward(ConformerGen* env) {
    if (SKIP_TERMINAL_MMFF_REWARD || env->num_accepted == 0) {
        return 0.0f;
    }

    double reward_start_ms = monotonic_ms();
    double mmff_energies[MAX_PROPOSALS];
    chem_mol_minimize_mmff_all_conformers(
        env->mol_handle,
        MMFF_MAX_ITERS,
        env->accepted_conf_ids,
        env->num_accepted,
        mmff_energies);

    double reward = 0.0;
    for (int i = 0; i < env->num_accepted; i++) {
        reward += exp(-((mmff_energies[i] - env->e0) / env->tau)) / env->z0;
    }

    env->episode_mmff_reward += reward;
    env->episode_mmff_reward_ms += monotonic_ms() - reward_start_ms;
    return (float)reward;
}

// ! TODO: Eval/render still needs TFD-based uniqueness; do not add TFD to training c_step.
void c_step(ConformerGen* env) {
    double step_start_ms = monotonic_ms();
    if (env->client && env->pending_reset) {
        c_reset(env);
        env->episode_step_ms += monotonic_ms() - step_start_ms;
        return;
    }

    if (env->mol_handle == NULL || env->num_molecules == 0) {
        fprintf(stderr, "conformer_gen: step called without a loaded molecule\n");
        abort();
    }
    
    float step_reward = apply_action(env);
    env->episode_length++;
    bool terminal = env->episode_length >= MAX_STEPS || env->num_proposed == MAX_PROPOSALS;

    if (terminal) {
        step_reward += terminal_mmff_reward(env);
    }

    env->rewards[0] = step_reward;
    env->terminals[0] = terminal ? 1.0f : 0.0f;
    env->episode_return += step_reward;
    compute_observations(env);
    env->episode_step_ms += monotonic_ms() - step_start_ms;

    if (terminal) {
        add_log(env);
        if (env->client) {
            env->pending_reset = 1;
        } else {
            c_reset(env);
        }
    }
}

static Vector3 v3_add(Vector3 a, Vector3 b) {
    return (Vector3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static Vector3 v3_sub(Vector3 a, Vector3 b) {
    return (Vector3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static Vector3 v3_scale(Vector3 v, float s) {
    return (Vector3){v.x * s, v.y * s, v.z * s};
}

static float v3_len(Vector3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

static Vector3 v3_cross(Vector3 a, Vector3 b) {
    return (Vector3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

static Vector3 v3_norm(Vector3 v) {
    float len = v3_len(v);
    if (len <= 1e-6f) {
        return (Vector3){0.0f, 1.0f, 0.0f};
    }
    return v3_scale(v, 1.0f / len);
}

static void reset_render_eval_metrics(Client* client, ConformerGen* env) {
    client->seen_episode_length = env->episode_length;
    client->seen_proposed = env->num_proposed;
    client->seen_accepted = env->num_accepted;
    client->last_status = 0;
    client->last_mmff_energy = 0.0;
    client->last_gibbs = 0.0;
    client->last_log_gibbs = 0.0;
    client->proposed_mmff_sum = 0.0;
    client->proposed_gibbs_sum = 0.0;
    client->proposed_log_gibbs_sum = 0.0;
    client->accepted_mmff_sum = 0.0;
    client->accepted_gibbs_sum = 0.0;
    client->accepted_log_gibbs_sum = 0.0;
}

static void update_render_camera(Client* client) {
    float ce = cosf(client->camera_elevation);
    client->camera.target = (Vector3){0.0f, 0.0f, 0.0f};
    client->camera.position = (Vector3){
        client->camera_distance * ce * sinf(client->camera_azimuth),
        client->camera_distance * sinf(client->camera_elevation),
        client->camera_distance * ce * cosf(client->camera_azimuth),
    };
}

static Client* make_conformer_client(ConformerGen* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    if (client == NULL) {
        fprintf(stderr, "conformer_gen: failed to allocate render client\n");
        abort();
    }

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(1120, 900, "PufferLib Conformer Gen");
    SetTargetFPS(RENDER_TARGET_FPS);

    client->camera_distance = 16.0f;
    client->camera_azimuth = 0.35f;
    client->camera_elevation = 0.35f;
    client->camera.up = (Vector3){0.0f, 1.0f, 0.0f};
    client->camera.fovy = 40.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    client->topology_molecule_index = -1;
    reset_render_eval_metrics(client, env);
    update_render_camera(client);
    return client;
}

static void handle_conformer_camera_controls(Client* client) {
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = true;
        client->last_mouse_pos = GetMousePosition();
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = false;
    }
    if (client->is_dragging) {
        Vector2 mouse = GetMousePosition();
        Vector2 delta = {mouse.x - client->last_mouse_pos.x, mouse.y - client->last_mouse_pos.y};
        client->camera_azimuth -= delta.x * 0.006f;
        client->camera_elevation += delta.y * 0.006f;
        client->camera_elevation = clamp_float(client->camera_elevation, -1.35f, 1.35f);
        client->last_mouse_pos = mouse;
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        client->camera_distance *= powf(0.88f, wheel);
        client->camera_distance = clamp_float(client->camera_distance, 3.0f, 120.0f);
    }
    if (IsKeyPressed(KEY_R)) {
        client->camera_azimuth = 0.35f;
        client->camera_elevation = 0.35f;
        client->camera_distance = fmaxf(8.0f, client->molecule_radius * 3.0f);
    }

    update_render_camera(client);
}

static void load_render_topology(ConformerGen* env, Client* client) {
    if (client->topology_molecule_index == env->molecule_index && client->num_atoms > 0) {
        return;
    }

    int atoms = 0;
    int bonds = 0;
    chem_mol_get_render_topology(
        env->mol_handle, client->atomic_numbers, MAX_RENDER_ATOMS, &atoms,
        client->bond_begin, client->bond_end, client->bond_order, client->bond_flags,
        MAX_RENDER_BONDS, &bonds);

    client->num_atoms = atoms;
    client->num_bonds = bonds;
    client->topology_molecule_index = env->molecule_index;
    reset_render_eval_metrics(client, env);
}

static void update_render_coords(ConformerGen* env, Client* client) {
    memset(client->coords, 0, sizeof(client->coords));
    int written = 0;
    chem_mol_get_render_coords(env->mol_handle, &client->coords[0][0], MAX_RENDER_ATOMS, &written);
    if (written != client->num_atoms || written <= 0) {
        fprintf(stderr, "conformer_gen: render coordinate count mismatch: topology=%d coords=%d\n",
            client->num_atoms, written);
        abort();
    }

    Vector3 center = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < client->num_atoms; i++) {
        center.x += client->coords[i][0];
        center.y += client->coords[i][1];
        center.z += client->coords[i][2];
    }
    center = v3_scale(center, 1.0f / (float)client->num_atoms);

    float radius = 1.0f;
    for (int i = 0; i < client->num_atoms; i++) {
        Vector3 p = {
            client->coords[i][0] - center.x,
            client->coords[i][1] - center.y,
            client->coords[i][2] - center.z,
        };
        client->coords[i][0] = p.x;
        client->coords[i][1] = p.y;
        client->coords[i][2] = p.z;
        radius = fmaxf(radius, v3_len(p));
    }

    if (client->molecule_radius <= 1.0f) {
        client->camera_distance = fmaxf(8.0f, radius * 3.0f);
    }
    client->molecule_radius = radius;
}

static void set_render_current_metric(Client* client, int status, double mmff_energy, double log_gibbs) {
    double gibbs = exp(log_gibbs);
    client->last_mmff_energy = mmff_energy;
    client->last_log_gibbs = log_gibbs;
    client->last_gibbs = gibbs;
    client->last_status = status;
}

static void add_render_proposal_metric(Client* client, bool accepted, double mmff_energy, double log_gibbs) {
    double gibbs = exp(log_gibbs);
    client->proposed_mmff_sum += mmff_energy;
    client->proposed_gibbs_sum += gibbs;
    client->proposed_log_gibbs_sum += log_gibbs;
    if (accepted) {
        client->accepted_mmff_sum += mmff_energy;
        client->accepted_gibbs_sum += gibbs;
        client->accepted_log_gibbs_sum += log_gibbs;
    }
}

static void compute_render_mmff_score(
        ConformerGen* env, int source_conf_id, double* out_mmff_energy, double* out_log_gibbs) {
    ChemMol* score_mol = NULL;
    chem_mol_clone_conformer(env->mol_handle, source_conf_id, &score_mol);
    chem_mol_minimize_mmff_conf(score_mol, source_conf_id, MMFF_MAX_ITERS);

    double mmff_energy = 0.0;
    chem_mol_energy_mmff(score_mol, source_conf_id, &mmff_energy);
    double log_gibbs = -((mmff_energy - env->e0) / env->tau) - log(env->z0);
    *out_mmff_energy = mmff_energy;
    *out_log_gibbs = log_gibbs;
    chem_mol_free(score_mol);
}

static void score_render_proposal(ConformerGen* env, Client* client, int source_conf_id, bool accepted) {
    double mmff_energy = 0.0;
    double log_gibbs = 0.0;
    compute_render_mmff_score(env, source_conf_id, &mmff_energy, &log_gibbs);
    set_render_current_metric(client, accepted ? 1 : 2, mmff_energy, log_gibbs);
    add_render_proposal_metric(client, accepted, mmff_energy, log_gibbs);
}

static void update_render_eval_scores(ConformerGen* env, Client* client) {
    if (env->episode_length < client->seen_episode_length ||
            env->num_proposed < client->seen_proposed ||
            env->num_accepted < client->seen_accepted) {
        reset_render_eval_metrics(client, env);
        return;
    }

    int proposed_delta = env->num_proposed - client->seen_proposed;
    int accepted_delta = env->num_accepted - client->seen_accepted;
    int rejected_delta = proposed_delta - accepted_delta;
    if (proposed_delta > 0) {
        for (int i = 0; i < accepted_delta; i++) {
            int accepted_idx = client->seen_accepted + i;
            if (accepted_idx >= 0 && accepted_idx < env->num_accepted) {
                score_render_proposal(env, client, env->accepted_conf_ids[accepted_idx], true);
            }
        }

        for (int i = 0; i < rejected_delta; i++) {
            score_render_proposal(env, client, env->working_conf_id, false);
        }
    }

    client->seen_episode_length = env->episode_length;
    client->seen_proposed = env->num_proposed;
    client->seen_accepted = env->num_accepted;
}

static Color atom_color(int atomic_number) {
    switch (atomic_number) {
        case 1: return (Color){248, 248, 248, 255};
        case 6: return (Color){86, 88, 90, 255};
        case 7: return (Color){38, 75, 210, 255};
        case 8: return (Color){224, 35, 35, 255};
        case 9: return (Color){96, 205, 90, 255};
        case 15: return (Color){235, 145, 35, 255};
        case 16: return (Color){220, 185, 45, 255};
        case 17: return (Color){90, 190, 70, 255};
        case 35: return (Color){125, 75, 45, 255};
        case 53: return (Color){105, 55, 150, 255};
        default: return (Color){155, 155, 155, 255};
    }
}

static float atom_radius(int atomic_number) {
    switch (atomic_number) {
        case 1: return 0.15f;
        case 6: return 0.26f;
        case 7: return 0.27f;
        case 8: return 0.27f;
        case 9: return 0.25f;
        case 15: return 0.34f;
        case 16: return 0.32f;
        case 17: return 0.32f;
        default: return 0.29f;
    }
}

static Vector3 render_atom_position(Client* client, int atom_idx) {
    return (Vector3){client->coords[atom_idx][0], client->coords[atom_idx][1], client->coords[atom_idx][2]};
}

static void draw_bond_cylinder(Vector3 a, Vector3 b, float radius, Color color) {
    if (v3_len(v3_sub(b, a)) > 1e-5f) {
        DrawCylinderEx(a, b, radius, radius, 12, color);
    }
}

static void draw_dashed_bond(Vector3 a, Vector3 b, float radius, Color color) {
    const int segments = 9;
    Vector3 delta = v3_sub(b, a);
    for (int i = 0; i < segments; i += 2) {
        Vector3 p0 = v3_add(a, v3_scale(delta, (float)i / (float)segments));
        Vector3 p1 = v3_add(a, v3_scale(delta, (float)(i + 1) / (float)segments));
        draw_bond_cylinder(p0, p1, radius, color);
    }
}

static void draw_render_bond(Client* client, int bond_idx) {
    int a_idx = client->bond_begin[bond_idx];
    int b_idx = client->bond_end[bond_idx];
    if (a_idx < 0 || a_idx >= client->num_atoms || b_idx < 0 || b_idx >= client->num_atoms) {
        return;
    }

    Vector3 a = render_atom_position(client, a_idx);
    Vector3 b = render_atom_position(client, b_idx);
    Vector3 dir = v3_norm(v3_sub(b, a));
    Vector3 offset = v3_cross(dir, client->camera.up);
    if (v3_len(offset) < 1e-4f) {
        offset = v3_cross(dir, (Vector3){1.0f, 0.0f, 0.0f});
    }
    offset = v3_scale(v3_norm(offset), 0.085f);

    Color bond_color = (Color){48, 48, 48, 255};
    int order = client->bond_order[bond_idx];
    bool aromatic = (client->bond_flags[bond_idx] & RENDER_BOND_AROMATIC) != 0;

    if (aromatic) {
        draw_bond_cylinder(a, b, 0.045f, bond_color);
        draw_dashed_bond(v3_add(a, offset), v3_add(b, offset), 0.028f, (Color){80, 80, 80, 255});
    } else if (order >= 3) {
        draw_bond_cylinder(a, b, 0.035f, bond_color);
        draw_bond_cylinder(v3_add(a, offset), v3_add(b, offset), 0.035f, bond_color);
        draw_bond_cylinder(v3_sub(a, offset), v3_sub(b, offset), 0.035f, bond_color);
    } else if (order == 2) {
        draw_bond_cylinder(v3_add(a, offset), v3_add(b, offset), 0.04f, bond_color);
        draw_bond_cylinder(v3_sub(a, offset), v3_sub(b, offset), 0.04f, bond_color);
    } else {
        draw_bond_cylinder(a, b, 0.05f, bond_color);
    }
}

static void draw_molecule(Client* client) {
    for (int i = 0; i < client->num_bonds; i++) {
        draw_render_bond(client, i);
    }
    for (int i = 0; i < client->num_atoms; i++) {
        int atomic_number = client->atomic_numbers[i];
        Vector3 p = render_atom_position(client, i);
        DrawSphere(p, atom_radius(atomic_number), atom_color(atomic_number));
        if (atomic_number == 1) {
            DrawSphereWires(p, atom_radius(atomic_number), 8, 8, (Color){150, 150, 150, 255});
        }
    }
}

static void draw_hud_text(const char* text, int x, int y, float size, Color color) {
    DrawText(text, x, y, (int)size, color);
}

static void draw_render_hud(ConformerGen* env, Client* client) {
    const Color text = (Color){28, 31, 34, 255};
    const Color muted = (Color){95, 101, 105, 255};
    const Color accent = (Color){57, 115, 77, 255};
    DrawRectangle(10, 8, 1100, 170, (Color){255, 255, 255, 220});
    DrawRectangleLines(10, 8, 1100, 170, (Color){185, 205, 190, 255});

    const int top = 18;
    const int line = 26;
    const int general_x = 22;
    const int proposed_x = 390;
    const int accepted_x = 745;
    const char* status = client->last_status == 1 ? "accepted" :
        (client->last_status == 2 ? "rejected" : (client->last_status == 3 ? "edit" : "none"));

    draw_hud_text("CURRENT", general_x, top, 22.0f, accent);
    draw_hud_text(TextFormat("mol %-4d step %-5d", env->molecule_index, env->episode_length),
        general_x, top + line, 20.0f, text);
    draw_hud_text(TextFormat("last %-10s", status), general_x, top + 2 * line, 20.0f, text);
    draw_hud_text(TextFormat("MMFF %.4f", client->last_mmff_energy),
        general_x, top + 3 * line, 20.0f, muted);
    draw_hud_text(TextFormat("Gibbs %.6g", client->last_gibbs),
        general_x, top + 4 * line, 20.0f, muted);
    draw_hud_text(TextFormat("logG %.6f", client->last_log_gibbs),
        general_x, top + 5 * line, 20.0f, muted);

    draw_hud_text("PROPOSED", proposed_x, top, 22.0f, accent);
    draw_hud_text(TextFormat("count %-4d", env->num_proposed),
        proposed_x, top + line, 20.0f, text);
    draw_hud_text(TextFormat("rejected %-4d", env->num_proposed - env->num_accepted),
        proposed_x, top + 2 * line, 20.0f, text);
    draw_hud_text(TextFormat("MMFF sum %.3f", client->proposed_mmff_sum),
        proposed_x, top + 3 * line, 20.0f, muted);
    draw_hud_text(TextFormat("Gibbs sum %.6g", client->proposed_gibbs_sum),
        proposed_x, top + 4 * line, 20.0f, muted);
    draw_hud_text(TextFormat("log sum %.6f", client->proposed_log_gibbs_sum),
        proposed_x, top + 5 * line, 20.0f, muted);

    draw_hud_text("ACCEPTED", accepted_x, top, 22.0f, accent);
    draw_hud_text(TextFormat("count %-4d", env->num_accepted),
        accepted_x, top + line, 20.0f, text);
    draw_hud_text(TextFormat("MMFF sum %.3f", client->accepted_mmff_sum),
        accepted_x, top + 2 * line, 20.0f, muted);
    draw_hud_text(TextFormat("Gibbs sum %.6g", client->accepted_gibbs_sum),
        accepted_x, top + 3 * line, 20.0f, muted);
    draw_hud_text(TextFormat("log sum %.6f", client->accepted_log_gibbs_sum),
        accepted_x, top + 4 * line, 20.0f, muted);
}

// Rendering done here
void c_close(ConformerGen* env) {
    if (env->client != NULL) {
        if (IsWindowReady()) {
            CloseWindow();
        }
        free(env->client);
        env->client = NULL;
    }
    deallocate(env);
}

void c_render(ConformerGen* env) {
    if (env->client == NULL) {
        env->client = make_conformer_client(env);
    }
    Client* client = env->client;

    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }
    if (env->mol_handle == NULL) {
        return;
    }

    load_render_topology(env, client);
    update_render_eval_scores(env, client);
    chem_mol_set_active_conformer(env->mol_handle, env->working_conf_id);
    update_render_coords(env, client);
    handle_conformer_camera_controls(client);

    BeginDrawing();
    ClearBackground((Color){250, 250, 249, 255});
    BeginMode3D(client->camera);
    draw_molecule(client);
    EndMode3D();
    draw_render_hud(env, client);
    EndDrawing();
}
