import gym
from gym import error, spaces, utils
from gym.utils import seeding

class AdvectionDiffusionFieldEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    ...

  def step(self, action):
    ...

  def reset(self):
    print("Reset")
    
  def render(self, mode='human', close=False):
    print("Render")