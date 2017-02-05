import numpy as np
from math import sqrt, log
from numpy import argmax

class Bandit_UCB:

    def __init__(self, num_arms):
        self._num_arms = num_arms

        self._arms = [i for i in range(self._num_arms)]

        self._means = [0 for i in range(self._num_arms)]
        self._preferences = [0 for i in range(self._num_arms)]
        self._round = 0

        self._reward_totals = [0 for i in range(self._num_arms)]
        self._choice_totals = [0 for i in range(self._num_arms)]

    def get_choice(self):
        if self._round < self._num_arms:
            return self._round
        else:
            return argmax(self._preferences)

    def update(self, arm, reward):
        self._reward_totals[arm] += reward
        self._choice_totals[arm] += 1
        self._round += 1
        self._calc_preferences(arm, reward)

    def _calc_preferences(self, arm, reward):
        self._means[arm] = self._reward_totals[arm] / self._choice_totals[arm]
        self._preferences[arm] += self._means[arm] + sqrt(2*log(self._round)/self._choice_totals[arm])


