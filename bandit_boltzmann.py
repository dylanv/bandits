import numpy as np


class BanditBoltzmann:

    def __init__(self, num_arms, temp=0.1):
        self._num_arms = num_arms
        self._temp = temp

        self._arms = [i for i in range(self._num_arms)]

        self._means = np.zeros(self._num_arms) + 0.5

        self._probabilities = np.zeros(self._num_arms)
        self._calc_probabilities()

        self._reward_totals = np.zeros(self._num_arms)
        self._choice_totals = np.zeros(self._num_arms)

    def get_choice(self):
        return np.random.choice(self._arms, p=self._probabilities)

    def update(self, arm, reward):
        self._reward_totals[arm] += reward
        self._choice_totals[arm] += 1
        self._means[arm] = self._reward_totals[arm] / self._choice_totals[arm]
        self._calc_probabilities()

    def _calc_probabilities(self):
        denom_sum = 0
        for j in range(self._num_arms):
            denom_sum += np.exp(self._means[j] / self._temp)

        total = 0
        for i in range(self._num_arms):
            self._probabilities[i] = np.exp(self._means[i] / self._temp) / denom_sum
            total += self._probabilities[i]

        for i in range(self._num_arms):
            self._probabilities[i] = self._probabilities[i] / total

    def get_probabilities(self):
        return self._probabilities
