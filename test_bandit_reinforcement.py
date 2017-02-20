from bandit_reinforcement import BanditReinforcement
import numpy as np


def test_reinforcement_bandit():
    reinforcement_bandit = BanditReinforcement(3)

    means = [0.3, 0.5, 0.7]
    std_devs = [0.1, 0.1, 0.1]

    for turn in range(10):
        choice = reinforcement_bandit.get_choice()
        reward = np.clip(np.random.normal(means[choice], std_devs[choice]), 0, 1)
        reinforcement_bandit.update(choice, reward)

    assert(reinforcement_bandit.get_probabilities().argmax() == 2)
