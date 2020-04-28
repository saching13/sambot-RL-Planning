import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .experiment import Experiment
import numpy as np

class AdvectionDiffusionFieldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 field_size=[100, 100],
                 field_vel=[-0.8, 0.8],
                 grid_size=[0.8, 0.8],
                 init_position=[10, 10],
                 dest_position=[90, 90]):

        self.experiment = Experiment(
            field_size=field_size,
            field_vel=field_vel,
            grid_size=grid_size,
            init_position=init_position,
            dest_position=dest_position)

        self.minFieldValue = 0
        self.maxFieldValue = 21

        self.min_r_x = 0
        self.max_r_x = 100
        
        self.min_r_y = 0
        self.max_r_y = 100
        
        self.min_Z_dot = -50
        self.max_Z_dot = 50

        self.min_z_grad_y = -1;
        self.max_z_grad_y = 1;

        self.min_z_grad_x = -1;
        self.max_z_grad_x = 1;
        
        # Define current position as initial position
        self.r = init_position

        # state vector = [r_x, r_y, z_r, z_grad_x, z_grad_y, z_dot]
        self.low = np.array([self.min_r_x, self.min_r_y, self.minFieldValue,  self.min_z_grad_x, self.min_z_grad_y, self.min_Z_dot], dtype = np.float32)
        self.high = np.array([self.max_r_x, self.max_r_y, self.maxFieldValue, self.max_z_grad_x, self.max_z_grad_y, self.max_Z_dot], dtype = np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.action_space = spaces.Discrete(4)


    def step(self, action):
        # state vector = [r_x, r_y, z_r, z_grad_x, z_grad_y, z_dot]

        # Ensure action is valid
        if action not in ["left", "right", "up", "down"]:
            print("Invalid action!")
            return False
        
        # Make a copy for the next location
        rK = self.r
        r_new = self.r

        #TODO(deepak): Add uncertainty to action
        # Calculate next location
        if action == "left":
            r_new[1] = r_new[1] - 1
        elif action == "right":
            r_new[1] = r_new[1] + 1
        elif action == "up":
            r_new[0] = r_new[0] - 1
        elif action == "down":
            r_new[0] = r_new[0] + 1

        # Check if done with learning
        done = True if r_new == self.experiment.dest_position else False
        
        
        # Update the field
        self.experiment.update_field()

        # Get the new state vector (observation) state vector = [r_x, r_y, z_r, z_grad_x, z_grad_y, z_dot]
        state_vector = self.experiment.get_state_vector(r_new)

        # TODO(Deepak): Figure out how to formulate reward
        self.experiment.updateViewScope(r_new)
        reward = self.experiment.reward(self.r, r_new)

        # Update the robot center location and append trajectory
        self.r = r_new
        self.trajectory.append(self.r)
        return done, state_vector, reward
        

    def reset(self):
        self.experiment.reset()
        state_vector = self.experiment.get_state_vector(self.r)
        return state_vector
    def render(self, mode='human', close=False):
        print("Render")
        # self.experiment.show_field_in_loop()
        self.experiment.show_field_state()

    def test_state(self):
        r = [2, 20]
        for k in range(50):
            state = self.experiment.get_state_vector(r)
            print(f"At time k = {k}, state = {state}")
            self.experiment.update_field()

