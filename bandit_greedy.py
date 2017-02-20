import numpy as np

class Bandit_Greedy:

    def __init__(self, num_arms, eps=0.1):
        self._num_arms = num_arms
        self._eps = eps

        self._means = np.zeros(self._num_arms) + 0.5
        self._arms = [i for i in range(self._num_arms)]
        self._probabilities = np.zeros(self._num_arms) + 1.0/self._num_arms
        self._reward_totals = np.zeros(self._num_arms)
        self._choice_totals = np.zeros(self._num_arms)

    def get_choice(self):
        return np.random.choice(self._arms, p=self._probabilities)

    def update(self, arm, reward):
        self._reward_totals[arm] += reward
        self._choice_totals[arm] += 1
        self._means[arm] = self._reward_totals[arm]/self._choice_totals[arm]
        max_arm = np.argmax(self._means)
        self._probabilities \
            = [1-self._eps if i==max_arm else self._eps/(self._num_arms-1) for i in range(self._num_arms)]