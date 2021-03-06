import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .experiment import Experiment
import numpy as np
import copy

class AdvectionDiffusionFieldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 field_size=[100, 100],
                 field_vel=[-0.8, 0.8],
                 grid_size=[0.8, 0.8],
                 init_position=[10, 10],
                 dest_position=[90, 90],
                 view_scope_size = 10):

        self.experiment = Experiment(
            field_size=field_size,
            field_vel=field_vel,
            grid_size=grid_size,
            init_position=init_position,
            dest_position=dest_position)

        self.min_field_value = 0
        self.max_field_value = 21

        self.min_r_x = 0
        self.max_r_x = 100
        
        self.min_r_y = 0
        self.max_r_y = 100
        
        self.min_z_dot = -7
        self.max_z_dot = 7

        self.min_z_grad_y = -1;
        self.max_z_grad_y = 1;

        self.min_z_grad_x = -1;
        self.max_z_grad_x = 1;
        self.max_Zdot = -float("inf")
        # Define current position as initial position
        self.r = copy.deepcopy(init_position)
        # self.offset = self.init_position - (view_scope_size // 2)

        # state vector = [r_x, r_y, z_r, z_grad_x, z_grad_y, z_dot]
        self.low = np.array([self.min_r_x, self.min_r_y, self.min_field_value,  self.min_z_grad_x, self.min_z_grad_y, self.min_z_dot], dtype = np.float32)
        self.high = np.array([self.max_r_x, self.max_r_y, self.max_field_value, self.max_z_grad_x, self.max_z_grad_y, self.max_z_dot], dtype = np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.action_space = spaces.Discrete(4)
        self.action_space_map = {}
        actions = ["left", "right", "up", "down"]
        for i, action in enumerate(actions):
            self.action_space_map[i] = action;

        # self.trajectory = []
        # self.trajectory.append(self.r)

    def step(self, action_id):
        # state vector = [r_x, r_y, z_r, z_grad_x, z_grad_y, z_dot]
        assert self.action_space.contains(action_id), "%r (%s) invalid" % (action_id, type(action_id))

        action = self.action_space_map[action_id]

        # Ensure action is valid
        assert action in ["left", "right", "up", "down"], "%s (%s) invalid" % (action, type(action))

        # Make a copy for the next location
        #r_k = self.r
        r_new = copy.deepcopy(self.r)

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
        self.max_Zdot = max(self.max_Zdot,  abs(state_vector[-1]))

        # print("Max Z dot : ", self.max_Zdot)

        # TODO(Deepak): Figure out how to formulate reward
        view_scope_state = self.experiment.update_view_scope(r_new)
        reward = self.experiment.reward(np.array(self.r), np.array(r_new))

        # Update the robot center location and append trajectory
        self.r = r_new
        # print("(x,y) :", r_new)
        self.experiment.trajectory.append(self.r)
        if not view_scope_state:
            done = True
            reward = -200
            print("-------------------> resetting due to out of bounds of view scope")
            print(self.experiment.trajectory)
            print("--------> printing out trajectory and ")

        print("[r_x, r_y, z_r, z_grad_x, z_grad_y, z_dot] :",  state_vector)

        # print("reward: ", reward)
        if r_new == self.experiment.dest_position: reward = 1000
        return state_vector, reward, done
        

    def reset(self):
        self.experiment.reset()
        self.r = self.experiment.init_position
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

