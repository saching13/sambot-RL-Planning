"""
q_learner.py
An easy-to-follow script to train, test and evaluate a Q-learning agent on the Mountain Car
problem using the OpenAI Gym. |Praveen Palanisamy
# Chapter 5, Hands-on Intelligent Agents with OpenAI Gym, 2018
"""
import gym
import numpy as np
from gym_adv_diff_field.envs.advection_diffusion_field_env import AdvectionDiffusionFieldEnv
import pickle

#MAX_NUM_EPISODES = 500
MAX_NUM_EPISODES = 70000
STEPS_PER_EPISODE = 200 #  This is specific to MountainCar. May change with env
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
NUM_DISCRETE_BINS = 50  # Number of bins to Discretize each observation dim

# Discretization parameters



class Q_Learner(object):
    def __init__(self, env):
        self.obs_shape = {2,}
        self.obs_high = env.observation_space.high[:2]
        self.obs_low = env.observation_space.low[:2]
        self.obs_bins = NUM_DISCRETE_BINS  # Number of bins to Discretize each observation dim
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n
        # Create a multi-dimensional array (aka. Table) to represent the
        # Q-values
        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1,
                           self.action_shape))  # (51 x 51 x 3)
        self.alpha = ALPHA  # Learning rate
        self.gamma = GAMMA  # Discount factor
        self.epsilon = 1.0

    def discretize(self, obs):
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        discretized_obs = self.discretize(obs)
        # Epsilon-Greedy action selection
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[discretized_obs])
        else:  # Choose a random action
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        print(np.max(self.Q[discretized_next_obs]))
        td_target = reward + self.gamma * np.max(self.Q[discretized_next_obs])
        td_error = td_target - self.Q[discretized_obs][action]
        self.Q[discretized_obs][action] += self.alpha * td_error


def train(agent, env):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = env.reset()
        obs = obs[:2]
        total_reward = 0.0
        if episode % 1000 == 0:
            print(episode, "---------->")
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done = env.step(action)
            next_obs = next_obs[:2]
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
    env = gym.make('adv-diff-field-v0',
                   field_size=[50, 50],
                   field_vel=[-0.2, 0.2],
                   grid_size=[0.8, 0.8],
                   dest_position=[40, 40],
                   init_position=[10, 10],
                   view_scope_size=11,
                   static_field=True)
    agent = Q_Learner(env)
    learned_policy = train(agent, env)
    # Use the Gym Monitor wrapper to evalaute the agent and record video
    # gym_monitor_path = "./gym_monitor_output"
    # env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    with open("learned_policy_div47.txt", 'wb') as policy_file:
        pickle.dump(learned_policy, policy_file)
    for _ in range(1000):
        test(agent, env, learned_policy)
    env.close()

