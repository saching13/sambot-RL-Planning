import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .experiment import Experiment

class AdvectionDiffusionFieldEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.experiment = Experiment(field_vel=[-2, 4])

  def step(self, action):
    ...

  def reset(self):
    print("Reset")
    
  def render(self, mode='human', close=False):
    print("Render")
    self.experiment.show_field_in_loop()
  
  def test_state(self):
    r = [2, 20]
    for k in range(50):
      state = self.experiment.get_state_vector(r)
      print(f"At time k = {k}, state = {state}")
      self.experiment.update_field()