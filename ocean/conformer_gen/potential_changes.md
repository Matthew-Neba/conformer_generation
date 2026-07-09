- see whether duplicating the rewards in a way by calculating thier finalized gibsbs score and then adding it to the rewards on terminal affects learning.Include , exclude them and see how learning is affected.

- see whether editing/removing these: #define PROPOSE_REWARD_BONUS 0.001f
#define DUPLICATE_PROPOSE_PENALTY (-PROPOSE_REWARD_BONUS)
#define OUT_OF_BOUNDS_EDIT_PENALTY -0.0002f helps learning

-see where Proposal observations can be stale after MMFF minimize (behavioral issue).
     propose_conformer() minimizes/changes the RDKit conformer, but does not refresh working_coords or current_torsion_bucket before compute_observations() runs in c_step.
     Relevant lines: pufferlib/ocean/conformer_gen/conformer_gen.h:678, pufferlib/ocean/conformer_gen/conformer_gen.h:683, pufferlib/ocean/conformer_gen/
     conformer_gen.h:774, and the observation sources pufferlib/ocean/conformer_gen/conformer_gen.h:472, pufferlib/ocean/conformer_gen/conformer_gen.h:488.
     Fix: call update_working_coords(env); and sync_torsion_buckets_from_conformer(env); in propose_conformer() after energy/minimize., see if implementing this idea helps learning or leaving it stale actually helps learning. (not sure which one is better )

-MMFF max iterations, MMFF intemediary iterations. Play around with tthat, want a good balance of low run time but accurate energy minimizations.
