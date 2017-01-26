import numpy as np

class Bandit_Boltzmann:

    def __init__(self, num_arms, eps=0.1):
        self._num_arms = num_arms
        self._arms = [i for i in range(self._num_arms)]
        self._reward_totals = [0 for i in range(self._num_arms)]
        self._choice_totals = [0 for i in range(self._num_arms)]

    def get_choice(self):
        return 0

    def update(self, arm, reward):
        return 0