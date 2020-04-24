import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .experiment import Experiment

class AdvectionDiffusionFieldEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.experiment = Experiment()

  def step(self, action):
    ...

  def reset(self):
    print("Reset")
    
  def render(self, mode='human', close=False):
    print("Render")
    self.experiment.show_curr_field()