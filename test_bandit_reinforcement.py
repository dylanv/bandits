from bandit_pursuit import Bandit_Pursuit
import numpy as np

pursuit_bandit = Bandit_Pursuit(3, learning_rate=0.5)

means = [0.3,0.5,0.7]
std_devs = [0.1, 0.1, 0.1]

for turn in range(1000):
    choice = pursuit_bandit.get_choice()
    reward = np.clip(np.random.normal(means[choice],std_devs[choice]),0,1)
    pursuit_bandit.update(choice, reward)

print(pursuit_bandit._probabilities)
