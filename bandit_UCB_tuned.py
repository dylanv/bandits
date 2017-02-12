import numpy as np

class Bandit_UCB_tuned:

    def __init__(self, num_arms):
        self._num_arms = num_arms

        self._means = np.zeros(self._num_arms)
        self._sum_of_squares = np.zeros(self._num_arms)
        self._preferences = np.zeros(self._num_arms)
        self._V = np.zeros(self._num_arms)
        self._round = 0

        self._reward_totals = np.zeros(self._num_arms)
        self._choice_totals = np.zeros(self._num_arms)

    def get_choice(self):
        if self._round < self._num_arms:
            return self._round
        else:
            return np.argmax(self._preferences)

    def update(self, arm, reward):
        self._round += 1
        self._calc_preferences(arm, reward)

    def _calc_preferences(self, arm, reward):
        self._reward_totals[arm] += reward
        self._choice_totals[arm] += 1

        self._means[arm] = self._reward_totals[arm] / self._choice_totals[arm]
        self._sum_of_squares[arm] += np.abs(reward - self._means[arm])**2

        denom = self._choice_totals[arm] - 1 if self._choice_totals[arm] - 1 >= 2 else self._choice_totals[arm]
        self._V[arm] = self._sum_of_squares[arm]/(denom)
        self._V[arm] += np.sqrt((2*np.log(self._round))/(self._choice_totals[arm]))

        self._preferences[arm] = self._means[arm]
        self._preferences[arm] += np.sqrt((2*np.log(self._round)/self._choice_totals[arm])*np.minimum(0.25,self._V[arm]))


