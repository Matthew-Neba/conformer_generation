#include "conformer_gen.h"

#define NUM_ATNS 1
#define ACT_SIZES {NUM_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define Env ConformerGen
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->owns_buffers = 0;
    env->client = NULL;
    env->mol_handle = NULL;
    env->molecule_index = -1;
    env->working_conf_id = -1;
    env->pending_reset = 0;

    env->molecule_family = (int)dict_get(kwargs, "molecule_family")->value;
    load_conformer_molecules(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "proposed", log->proposed);
    dict_set(out, "accepted", log->accepted);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "mmff_reward", log->mmff_reward);
    dict_set(out, "mmff_reward_ms", log->mmff_reward_ms);
    dict_set(out, "duplicate_check_ms", log->duplicate_check_ms);
    dict_set(out, "set_torsion_ms", log->set_torsion_ms);
    dict_set(out, "coords_ms", log->coords_ms);
    dict_set(out, "obs_ms", log->obs_ms);
    dict_set(out, "copy_conf_ms", log->copy_conf_ms);
    dict_set(out, "sync_torsion_ms", log->sync_torsion_ms);
    dict_set(out, "load_selected_ms", log->load_selected_ms);
    dict_set(out, "propose_ms", log->propose_ms);
    dict_set(out, "apply_action_ms", log->apply_action_ms);
    dict_set(out, "step_ms", log->step_ms);
    dict_set(out, "accounted_ms", log->accounted_ms);
    dict_set(out, "reset_init_ms", log->reset_init_ms);
}
