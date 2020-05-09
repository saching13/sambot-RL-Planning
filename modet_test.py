import gym
from gym_adv_diff_field.envs.advection_diffusion_field_env import AdvectionDiffusionFieldEnv
import numpy as np
import collections
import warnings
import pickle
# import keras.backend.tensorflow_backend as backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
from tqdm import tqdm
import random
import tensorflow as tf

# new_model = tf.keras.models.load_model('models/SamBot__best_reward__-81165.06.model')
new_model = tf.keras.models.load_model('SamBot_final.model')

print(new_model.summary())