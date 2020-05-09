"""
q_learner.py
An easy-to-follow script to train, test and evaluate a Q-learning agent on the Mountain Car
problem using the OpenAI Gym. | Praveen Palanisamy
# Chapter 5, Hands-on Intelligent Agents with OpenAI Gym, 2018
"""
import gym
from gym_adv_diff_field.envs.advection_diffusion_field_env import AdvectionDiffusionFieldEnv
import numpy as np
import pickle

# MAX_NUM_EPISODES = 500
MAX_NUM_EPISODES = 50000
NUM_TEST_EPISODES = 1000
STEPS_PER_EPISODE = 300  # This is specific to MountainCar. May change with env
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
SCALAR_FIELD = True
# NUM_DISCRETE_BINS = 100  # Number of bins to Discretize each observation dim


discreteBins_r = 50
discreteBins_FieldValue = 420
discreteBins_z_grad = 20
discreteBins_z_dot = 300


class Q_Learner(object):
    def __init__(self, env, scalar_field=False):
        self.scalar_field = scalar_field
        # self.obs_shape = env.observation_space.shape
        self.obs_shape = (2,)
        self.obs_high = env.observation_space.high[:2]
        self.obs_low = env.observation_space.low[:2]
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
        # binWidth_fieldValue = space[2] / discreteBins_FieldValue
        # binWidth_z_grad = space[3] / discreteBins_z_grad

        if not self.scalar_field:
            binWidth_z_dot = space[5] / discreteBins_z_dot
            return [binWidth_r, binWidth_r]
        else:
            return [binWidth_r, binWidth_r]

    def discretize_state_vector(self, state_vector):
        # state vector = [r_x, r_y, z_r, z_grad_x, z_grad_y, z_dot]
        obs = []
        for i in range(self.obs_shape[0]):
            obs.append(self.discretize(state_vector[i], self.obs_low[i], self.bin_width[i]))
        return obs

    def discretize(self, obs, low, binWidth):
        return ((obs - low) / binWidth).astype(int)

    def get_max(self, discretized_obs, arg):
        # print(type(self.Q))
        # actions = self.Q[discretized_obs] if discretized_obs in self.Q else {}
        try:
            actions = self.Q[tuple(discretized_obs)]
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
        except KeyError:
            if arg:
                # return np.random.choice([a for a in range(self.action_shape)])
                return 0
            else:
                return 0

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
        # print("Zdot bin:", discretized_obs[-1])
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
    return create_policy(agent)
    
def create_policy(agent):
    policy = {}
    # print(type(action_dict))

    for state, action_dict in agent.Q.items():
        best_action = 0
        best_Q_value = -float("inf")
        for action, Q_value in action_dict.items():
            if Q_value > best_Q_value:
                best_Q_value = Q_value
                best_action = action
        
        policy[tuple(state)] = best_action
    print(type(policy))
    return policy

def get_policy_action(agent, obs, policy):
    discritized_obs = agent.discretize_state_vector(obs)
    try:
        return policy[tuple(discritized_obs)]
    except KeyError:
        # warnings.warn("state not found in Policy", KeyError)
        print("state not found in Policy")
        return 0;

def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        action = get_policy_action(agent, obs, policy)
        next_obs, reward, done = env.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward


if __name__ == "__main__":
    
    # Dynamic Field
    if not SCALAR_FIELD:
        env = gym.make('adv-diff-field-v0',
                    field_size=[100, 100],
                    field_vel=[-0.2, 0.2],
                    grid_size=[0.8, 0.8],
                    dest_position=[95, 95],
                    init_position=[10, 10],
                    view_scope_size=11,
                    static_field=False)

# Static field
    else:
        env = gym.make('adv-diff-field-v0',
                field_size=[50, 50],
                field_vel=[-0.2, 0.2],
                grid_size=[0.8, 0.8],
                dest_position=[40, 40],
                init_position=[10, 10],
                view_scope_size=11,
                static_field=True)
    
    agent = Q_Learner(env, scalar_field=SCALAR_FIELD)
    learned_policy = train(agent, env)
    with open("learned_policy_without_minus_30.txt", 'wb') as policy_file:
        pickle.dump(learned_policy, policy_file)
    # Use the Gym Monitor wrapper to evalaute the agent and record video
    # gym_monitor_path = "./gym_monitor_output"
    # env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    for episode in range(NUM_TEST_EPISODES):
        reward = test(agent, env, learned_policy)
        print("Test Episode#:{} reward:{}".format(episode,reward))
    env.close()
