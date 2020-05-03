import gym
import gym_adv_diff_field

import pickle
pol = pickle.load(open("learned_policy.txt",'rb'))
print(type(pol))
print(len(pol))
print(pol)