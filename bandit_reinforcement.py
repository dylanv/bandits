import numpy as np

class Bandit_Reinforcement:

    def __init__(self, num_arms, alpha=0.1, beta=0.1):
        self._num_arms = num_arms
        self._alpha= alpha
        self._beta = beta

        self._arms = [i for i in range(self._num_arms)]
        self._avg_reward = 0.5

        self._preferences = [1.0 / self._num_arms for i in range(self._num_arms)]
        self._probabilities = [1.0/self._num_arms for i in range(self._num_arms)]

        self._reward_totals = [0 for i in range(self._num_arms)]
        self._choice_totals = [0 for i in range(self._num_arms)]

    def get_choice(self):
        return np.random.choice(self._arms, p=self._probabilities)

    def update(self, arm, reward):
        self._reward_totals[arm] += reward
        self._choice_totals[arm] += 1
        self._calc_probabilities(arm, reward)

    def _calc_probabilities(self, arm, reward):
        self._preferences[arm] += self._beta*(reward - self._avg_reward)
        self._avg_reward = (1-self._alpha)*self._avg_reward + self._alpha*reward

