import gym
import gym_adv_diff_field

env = gym.make('adv-diff-field-v0')

env.reset()
# env.render()
env.test_state()