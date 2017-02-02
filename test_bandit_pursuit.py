from bandit_reinforcement import Bandit_Reinforcement
import numpy as np

reinforcement_bandit = Bandit_Reinforcement(3)

means = [0.3,0.5,0.7]
std_devs = [0.1, 0.1, 0.1]

for turn in range(1000):
    choice = reinforcement_bandit.get_choice()
    reward = np.clip(np.random.normal(means[choice],std_devs[choice]),0,1)
    reinforcement_bandit.update(choice, reward)

print(reinforcement_bandit._probabilities)

