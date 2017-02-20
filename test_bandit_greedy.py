from bandit_greedy import Bandit_Greedy
import numpy as np

def test_greedy_bandit():

    greedy_bandit = Bandit_Greedy(3)

    means = [0.3,0.5,0.7]
    std_devs = [0.1, 0.1, 0.1]
    choice_counts = {0:0,1:0,2:0}

    for turn in range(1000):
        choice = greedy_bandit.get_choice()
        reward = np.clip(np.random.normal(means[choice],std_devs[choice]),0,1)
        greedy_bandit.update(choice, reward)
        choice_counts[choice] += 1

    assert(greedy_bandit._means.argmax() == 2)
    assert (choice_counts[2] > choice_counts[1] and choice_counts[2] > choice_counts[0])