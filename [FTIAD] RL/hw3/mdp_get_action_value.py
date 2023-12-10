
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """

    Q = 0
    next_states_probs = mdp.get_next_states(state, action)
    for new_state_prob in next_states_probs:
      P = next_states_probs[new_state_prob]
      r = mdp.get_reward(state, action, new_state_prob)
      V = state_values[new_state_prob]
      Q += P * (r + gamma * V)

    return Q
