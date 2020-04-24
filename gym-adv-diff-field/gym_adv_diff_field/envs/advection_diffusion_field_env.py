import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .experiment import Experiment


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

        # Define current position as initial position
        self.r = init_position

    def step(self, action):
        # Ensure action is valid
        if action not in ["left", "right", "up", "down"]:
            print("Invalid action!")
            return False
        
        # Make a copy for the next location
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
        done = True if r_new == dest_position else False
        
        # TODO(Deepak): Figure out how to formulate reward
        reward = -1
        
        # Update the field
        self.experiment.update_field()

        # Get the new state vector (observation)
        state_vector = self.experiment.get_state_vector(r_new)

        # Update the robot center location and append trajectory
        self.r = r_new
        self.trajectory.append(self.r)
        return done, state_vector, reward
        

    def reset(self):
        self.experiment.reset()

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

