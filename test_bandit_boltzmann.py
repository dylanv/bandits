from bandit_boltzmann import Bandit_Boltzmann
import numpy as np

boltzmann_bandit = Bandit_Boltzmann(3, temp=0.01)

means = [0.3,0.5,0.7]
std_devs = [0.1, 0.1, 0.1]

for turn in range(1000):
    choice = boltzmann_bandit.get_choice()
    reward = np.clip(np.random.normal(means[choice],std_devs[choice]),0,1)
    boltzmann_bandit.update(choice, reward)

print(boltzmann_bandit._probabilities)