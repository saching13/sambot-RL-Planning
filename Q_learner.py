"""
q_learner.py
An easy-to-follow script to train, test and evaluate a Q-learning agent on the Mountain Car
problem using the OpenAI Gym. | Praveen Palanisamy
# Chapter 5, Hands-on Intelligent Agents with OpenAI Gym, 2018
"""
import gym
from gym_adv_diff_field.envs.advection_diffusion_field_env import AdvectionDiffusionFieldEnv
import numpy as np
import collections

# MAX_NUM_EPISODES = 500
MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200  # This is specific to MountainCar. May change with env
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
# NUM_DISCRETE_BINS = 100  # Number of bins to Discretize each observation dim


discreteBins_r = 100
discreteBins_FieldValue = 100
discreteBins_z_grad = 500
discreteBins_z_dot = 500


class Q_Learner(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        # self.obs_bins = NUM_DISCRETE_BINS  # Number of bins to Discretize each observation dim
        self.bin_width = self.create_obs_state_bins(self.obs_high - self.obs_low)
        self.action_shape = env.action_space.n
        # Create a multi-dimensional array (aka. Table) to represent the
        # Q-values

        self.Q = {}


        self.alpha = ALPHA  # Learning rate
        self.gamma = GAMMA  # Discount factor
        self.epsilon = 1.0

    def create_obs_state_bins(self, space):
        binWidth_r = space[0] / discreteBins_r
        binWidth_fieldValue = space[2] / discreteBins_FieldValue
        binWidth_z_grad = space[3] / discreteBins_z_grad

        binWidth_z_dot = space[5] / discreteBins_z_dot
        return [binWidth_r, binWidth_r, binWidth_fieldValue, binWidth_z_grad, binWidth_z_grad, binWidth_z_dot]

    def discretize_state_vector(self, stateVector):
        # state vector = [r_x, r_y, z_r, z_grad_x, z_grad_y, z_dot]
        obs = []
        for i in range(self.obs_shape[0]):
            obs.append(self.discretize(stateVector[i], self.obs_low[i], self.bin_width[i]))
        return obs

    def discretize(self, obs, low, binWidth):
        return ((obs - low) / binWidth).astype(int)

    def get_max(self, discretized_obs, arg):
        # print(type(self.Q))
        # actions = self.Q[discretized_obs] if discretized_obs in self.Q else {}
        try:
            actions = self.Q[tuple(discretized_obs)]
        except KeyError:
            return 0

        maxArg = 0
        maxValue = -float('inf')
        for action, value in actions.items():
            if value > maxValue:
                maxArg = action
                maxValue = value

        if arg:
            return maxArg
        else:
            return maxValue if maxValue != -float('inf') else 0

    def get_action(self, obs):
        discretized_obs = self.discretize_state_vector(obs)
        # Epsilon-Greedy action selection
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            return self.get_max(discretized_obs, True)
        else:  # Choose a random action
            return np.random.choice([a for a in range(self.action_shape)])

    def get_Q_value(self, stateVector, action):
        Q_action_dict = self.Q[tuple(stateVector)] if tuple(stateVector) in self.Q else {}
        return Q_action_dict[action] if action in Q_action_dict else 0

    def update_q(self, discretized_obs, action, td_error):
        try:
            self.Q[tuple(discretized_obs)][action] += self.alpha * td_error
        except KeyError:
            self.Q[tuple(discretized_obs)] = {action: self.alpha * td_error}

    def learn(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize_state_vector(obs)
        print("Zdot bin:", discretized_obs[-1])
        discretized_next_obs = self.discretize_state_vector(next_obs)
        td_target = reward + self.gamma * self.get_max(discretized_next_obs, False)
        td_error = td_target - self.get_Q_value(discretized_obs, action)
        self.update_q(discretized_obs, action, td_error)
        # self.Q[tuple(discretized_obs)][action] += self.alpha * td_error


def train(agent, env):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = env.reset()
        total_reward = 0.0
        if episode % 1000 == 0:
            print(episode, "---------->")
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
                                                                   total_reward, best_reward, agent.epsilon))
    # Return the trained policy
    return np.argmax(agent.Q, axis=2)


def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done = env.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward


if __name__ == "__main__":
    env = gym.make('adv-diff-field-v0')
    agent = Q_Learner(env)
    learned_policy = train(agent, env)
    # Use the Gym Monitor wrapper to evalaute the agent and record video
    gym_monitor_path = "./gym_monitor_output"
    env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    for _ in range(1000):
        test(agent, env, learned_policy)
    env.close()
