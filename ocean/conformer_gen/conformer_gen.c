#include "conformer_gen.h"

int main(void) {
    ConformerGen env;
    memset(&env, 0, sizeof(env));
    c_reset(&env);

    for (int i = 0; i < 8; i++) {
        if (i % 4 == 3) {
            env.actions[0] = (float)ACTION_PROPOSE;
        } else {
            env.actions[0] = (float)(i % NUM_TORSION_ACTIONS);
        }
        c_step(&env);
    }

    c_close(&env);
    return 0;
}
