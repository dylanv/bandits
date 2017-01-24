from bandit_greedy import Bandit_Greedy
import numpy as np

greedy_bandit = Bandit_Greedy(3)

means = [0.3,0.5,0.7]
std_devs = [0.1, 0.1, 0.1]

for turn in range(1000):
    choice = greedy_bandit.get_choice()
    reward = np.clip(np.random.normal(means[choice],std_devs[choice]),0,1)
    greedy_bandit.update(choice, reward)

print(greedy_bandit._means)