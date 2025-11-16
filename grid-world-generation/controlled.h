#pragma once
#include "mdp.h"

// Sutton & Barto Example 3.5 style small grids (3x3, 4x4, 5x5), deterministic, hand-crafted.
void generate_toy_grid(int n, MDPDense& out, double goal_reward=1.0);

// Edge cases
void generate_single_state(MDPDense& out);                  // one state, one action
void generate_disconnected(int S, int A, MDPDense& out);    // some states with zero outgoing prob
void generate_absorbing(int S, int A, MDPDense& out, bool reward_one);
void generate_zero_reward(int S, int A, MDPDense& out);     // all-zero rewards
void generate_high_gamma_cycle(int cycle_len, double gamma, MDPDense& out);
