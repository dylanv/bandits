from bandit_UCB import BanditUCB
import numpy as np


def test_ucb_bandit():
    ucb_bandit = BanditUCB(3)

    means = [0.3, 0.5, 0.7]
    std_devs = [0.1, 0.1, 0.1]

    for turn in range(1000):
        choice = ucb_bandit.get_choice()
        reward = np.clip(np.random.normal(means[choice], std_devs[choice]), 0, 1)
        ucb_bandit.update(choice, reward)

    assert(ucb_bandit.get_preferences().argmax() == 2)
