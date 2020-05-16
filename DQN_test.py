from keras.models import load_model
import tensorflow as tf 
import os
import numpy as np
import gym
from gym_adv_diff_field.envs.advection_diffusion_field_env import AdvectionDiffusionFieldEnv

def test(model, env,episode):
    done = False
    current_state = env.reset()
    total_reward = 0.0
    with tf.device('/GPU:0'):
        while not done:
            action = np.argmax(model.predict(np.array(np.asarray(current_state)).reshape(-1, *np.asarray(current_state).shape)))
            next_state, reward, done = env.step(action)
            current_state = next_state
            total_reward += reward
    return total_reward


if __name__ == "__main__":
    env = gym.make('adv-diff-field-v0',field_size=[50, 50],
    field_vel=[-0.2, 0.2],grid_size=[0.8, 0.8],dest_position=[40, 40],init_position=[10, 10],view_scope_size=11,static_field=True)
    #Loading the model
    path = os.path.join("/Users/jayaprakashreddydumpa/Desktop/sambot-RL-Planning","models/SamBot_final.model/")
    model = tf.keras.models.load_model(path)
    # print(model)
    for episode in range(100):
        reward = test(model,env,episode)
        print("Test Episode#:{} reward:{}".format(episode,reward))
