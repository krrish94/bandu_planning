from __future__ import print_function

from bandu_stacking.policies.policy import Policy


class RandomPolicy(Policy):
    def __init__(self, env):
        self.env = env
        super(RandomPolicy, self).__init__()

    def get_action(self, initial_state):
        return self.env.sample_action()
