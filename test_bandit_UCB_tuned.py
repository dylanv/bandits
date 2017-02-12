from bandit_UCB_tuned import Bandit_UCB_tuned
import numpy as np

UCB_tuned_bandit = Bandit_UCB_tuned(3)

means = [0.4,0.5,0.6]
std_devs = [0.1, 0.1, 0.1]

for turn in range(1000):
    choice = UCB_tuned_bandit.get_choice()
    reward = np.clip(np.random.normal(means[choice],std_devs[choice]),0,1)
    UCB_tuned_bandit.update(choice, reward)

print(UCB_tuned_bandit._preferences)
print(UCB_tuned_bandit._V)
print(UCB_tuned_bandit._sum_of_squares)
print(UCB_tuned_bandit._choice_totals)
