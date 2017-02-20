from bandit_pursuit import BanditPursuit
import numpy as np


def test_pursuit_bandit():

    pursuit_bandit = BanditPursuit(3, learning_rate=0.5)

    means = [0.3, 0.5, 0.7]
    std_devs = [0.1, 0.1, 0.1]

    for turn in range(1000):
        choice = pursuit_bandit.get_choice()
        reward = np.clip(np.random.normal(means[choice], std_devs[choice]), 0, 1)
        pursuit_bandit.update(choice, reward)

    assert(pursuit_bandit.get_probabilities().argmax() == 2)
