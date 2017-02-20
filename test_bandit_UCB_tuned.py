from bandit_UCB_tuned import BanditUCBtuned
import numpy as np


def test_ucb_tuned_bandit():
    ucb_tuned_bandit = BanditUCBtuned(3)

    means = [0.4, 0.5, 0.6]
    std_devs = [0.1, 0.1, 0.1]

    for turn in range(1000):
        choice = ucb_tuned_bandit.get_choice()
        reward = np.clip(np.random.normal(means[choice], std_devs[choice]), 0, 1)
        ucb_tuned_bandit.update(choice, reward)

    assert(ucb_tuned_bandit.get_preferences().argmax() == 2)
