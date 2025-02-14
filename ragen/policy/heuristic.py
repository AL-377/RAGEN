from .base import BasePolicy
import random

class FixedPolicy(BasePolicy):
    def select_action(self, observation, env=None):
        return random.choice(env.get_all_actions())
