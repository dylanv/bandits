from bandit_UCB import Bandit_UCB
import numpy as np

UCB_bandit = Bandit_UCB(3)

means = [0.3,0.5,0.7]
std_devs = [0.1, 0.1, 0.1]

for turn in range(1000):
    choice = UCB_bandit.get_choice()
    reward = np.clip(np.random.normal(means[choice],std_devs[choice]),0,1)
    UCB_bandit.update(choice, reward)

print(UCB_bandit._preferences)
