import numpy as np

class Bandit_Pursuit:

    def __init__(self, num_arms, learning_rate=0.5):

        self._num_arms = num_arms
        self._learning_rate = learning_rate

        self._arms = [i for i in range(self._num_arms)]
        self._means = [0.5 for i in range(self._num_arms)]

        self._probabilities = [1.0/num_arms for i in range(self._num_arms)]

        self._reward_totals = [0 for i in range(self._num_arms)]
        self._choice_totals = [0 for i in range(self._num_arms)]

    def get_choice(self):
        return np.random.choice(self._arms, p=self._probabilities)

    def update(self, arm, reward):
        self._reward_totals[arm] += reward
        self._choice_totals[arm] += 1
        self._means[arm] = self._reward_totals[arm] / self._choice_totals[arm]
        self._calc_probabilities()

    def _calc_probabilities(self):
        max_arm = np.argmax(self._means)
        for arm in range(self._num_arms):
            if arm == max_arm:
                self._probabilities[arm] += self._learning_rate * (1 - self._probabilities[arm])
            else:
                self._probabilities[arm] += self._learning_rate * (0 - self._probabilities[arm])
