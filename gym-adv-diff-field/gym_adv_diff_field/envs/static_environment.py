import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy import io
import os
import time
import copy

class StaticExperiment:

    def __init__(
            self,
            field_size=[50, 50],
            grid_size=[0.8, 0.8],
            init_position=[5, 5],
            dest_position=[42, 42],
            view_scope_size=5,
            weights=[1, 1, 10]):

        self.field_size = field_size
        # self.field_vel = field_vel
        self.dx = grid_size[0]
        self.dy = grid_size[1]
        self.dt = 0.1

        self.k1 = weights[0]
        self.k2 = weights[1]
        self.k3 = weights[2]

        self.view_scope_size = view_scope_size
        self.view_scope = np.zeros((self.view_scope_size, self.view_scope_size))
        # self.normalized_view_scope = np.zeros((self.view_scope_size, self.view_scope_size))
        self.init_position = copy.deepcopy(init_position)
        self.dest_position = dest_position

        self.agent_field_state = np.zeros(field_size)

        self.curr_field = self.create_field()
        self.prev_field = np.zeros(field_size)

        self.trajectory = []

        # Display
        # self.fig_field = plt.figure(figsize=(8, 8))
        # self.fig_field_ax = self.fig_field.add_subplot(111)
        # self.fig_field_ax.set_title('Field State')
        # self.fig_field_ax.set_aspect('equal')

    def set_init_position(self, init_position):
        if (init_position[0] < 0 or init_position[0] > self.field_size[0]) or (
                init_position[1] < 0 or init_position[1] > self.field_size[1]):
            print("[ERROR] Initial position out of range")
        self.init_position = init_position

    def set_dest_position(self, dest_position):
        if (dest_position[0] < 0 or dest_position[0] > self.field_size[0]) or (
                dest_position[1] < 0 or dest_position[1] > self.field_size[1]):
            print("[ERROR] Destination position out of range")
        self.dest_position = dest_position

    def reset(self):
        self.curr_field = self.create_field()
        self.prev_field = np.zeros(self.field_size)
        self.trajectory = [self.init_position]
        self.agent_field_state = np.zeros(self.field_size)
        print("Experiment Reset")

    def create_field(self):
        cwd = os.getcwd()

        combined_field = np.load(cwd + "/curr_field.npy")
        combined_field = combined_field[::2, ::2]

        return combined_field

    def zmf(self, x, a, b):
        mid = (a + b) / 2
        if (x < a):
            return 1
        elif (a <= x <= mid):
            return (1 - 2 * ((x - a) / (b - a)) ** 2)
        elif (mid < x <= b):
            return 2 * ((x - b) / (b - a)) ** 2
        else:
            return 0

    # def update_view_scope(self, r):
    #     """
    #     Copy the view scope directly to agent_field_state in the correct location
    #     no need to normalize
    #     def copy_view_scope
    #     """
    #     scopeRange = self.view_scope_size // 2;

    #     ## TODO: Is x is rows or y is rows. check this later

    #     min_index = [r[0] - scopeRange, r[1] - scopeRange]
    #     max_index = [r[0] + scopeRange, r[1] + scopeRange]

    #     if min_index[0] < 0 or min_index[1] < 0 or max_index[0] >= self.field_size[0] or max_index[1] > self.field_size[1]:
    #         return False

    #     self.view_scope = self.curr_field[r[0] - scopeRange: r[0] + scopeRange,
    #                       r[1] - scopeRange: r[1] + scopeRange].copy()
    #     maxVal = self.view_scope.max()
    #     minVal = self.view_scope.min()
    #     self.normalized_view_scope = (self.view_scope - minVal) / (maxVal - minVal)

    #     return True

    def normalize(self, field):
        max_val = field.max()
        min_val = field.min()
        field_normalized = (field - min_val) / (max_val - min_val)
        return field_normalized

    def copy_view_scope(self, r):
        scope_range = self.view_scope_size // 2;

        top_left_x = max(0, r[0] - scope_range)
        top_left_y = max(0, r[1] - scope_range)

        bottom_right_x = min(self.field_size[0] - 1, r[0] + scope_range)
        bottom_right_y = min(self.field_size[1] - 1, r[1] + scope_range)

        self.view_scope = self.curr_field[top_left_y : bottom_right_y + 1, top_left_x : bottom_right_x + 1]

        self.agent_field_state[top_left_y : bottom_right_y + 1, top_left_x : bottom_right_x + 1] = \
            self.curr_field[top_left_y : bottom_right_y + 1, top_left_x : bottom_right_x + 1]
    
    def get_normalized_view_scope_at(self, r):
        max_val = self.view_scope.max()
        min_val = self.view_scope.min()
        view_scope_normalized = (self.view_scope - min_val) / (max_val - min_val)
        index = self.view_scope_size // 2
        return view_scope_normalized[index, index] # center value of the view scope

    def calculate_mapping_error(self):
        return np.sum(self.normalize(np.abs(self.agent_field_state - self.curr_field)))
    
    # def calculate_reward(self, r):
    #     traj_len = len(self.trajectory)
    #     mapping_error = self.calculate_mapping_error()
    #     dist_to_goal = np.linalg.norm(np.array(r) - np.array(self.dest_position))

    #     # print(f"Rewards: {traj_len}, {mapping_error}, {dist_to_goal}")

    #     return -(self.k1 * traj_len + self.k2 * mapping_error + self.k3 * dist_to_goal)

    def get_normalized_field_value(self, field, r):
        max_val = field.max()
        min_val = field.min()
        field_value_r_normalized = (field[r[1], r[0]] - min_val) / (max_val - min_val)
        return field_value_r_normalized 

    def calculate_reward_1(self, r): # Option 2
        """
        1. Mapping error only measuremnet + ZMF
        """
        mapping_error = self.get_normalized_view_scope_at(r)
        zmf_error = self.zmf(mapping_error, 0, 1)
        dist_to_goal = np.linalg.norm(np.array(r) - np.array(self.dest_position))
        print(f"Rewards: {zmf_error}, {dist_to_goal}")
        return (self.k2 * zmf_error - self.k3 * dist_to_goal)
    
    def calculate_reward_2(self, r): # Option 2
        """
        2. Mapping error only measuremnet + negation
        """
        mapping_error = self.get_normalized_view_scope_at(r)
        dist_to_goal = np.linalg.norm(np.array(r) - np.array(self.dest_position))
        print(f"Rewards: {mapping_error}, {dist_to_goal}")
        return -(self.k2 * mapping_error + self.k3 * dist_to_goal)

    def calculate_reward_3(self, r): # Option 2
        """
        3. Mapping error y - z_prev(r) + negation
        """
        map_error = self.curr_field - self.prev_field
        field_error_at_r = self.get_normalized_field_value(map_error, r)
        dist_to_goal = np.linalg.norm(np.array(r) - np.array(self.dest_position))
        print(f"Rewards: {field_error_at_r}, {dist_to_goal}")
        return -(self.k2 * field_error_at_r + self.k3 * dist_to_goal)

    def calculate_reward_4(self, r): # Option 2
        """
        4. Mapping error y - z_prev(r) + zmf
        """
        map_error = self.curr_field - self.prev_field
        field_error_at_r = self.get_normalized_field_value(map_error, r)
        zmf_error = self.zmf(field_error_at_r, 0, 1)
        dist_to_goal = np.linalg.norm(np.array(r) - np.array(self.dest_position))
        print(f"Rewards: {zmf_error}, {dist_to_goal}")
        return (self.k2 * zmf_error - self.k3 * dist_to_goal)

    # def calculate_reward(self, r_k, r):
    #     offset = r - self.view_scope_size // 2
    #     view_scope_index = r_k - offset
    #     distance = self.k2 * np.linalg.norm(r - r_k) + \
    #                self.zmf(self.normalized_view_scope[view_scope_index[0],
    #                                                     view_scope_index[1]], 0, 1)
    #     return distance

    def reset_init_position(self):
        view_scope_offset = self.view_scope_size // 2
        r_x = np.random.randint(view_scope_offset, self.field_size[0] - view_scope_offset)
        r_y = np.random.randint(view_scope_offset, self.field_size[1] - view_scope_offset)
        self.init_position = [r_x, r_y]
        return [r_x, r_y]

    def update_field(self):
        pass

    def get_z_dot(self, r):
        pass

    def get_gradient(self, r):

        dz_dx = (self.curr_field[r[1], r[0] + 1] - self.curr_field[r[1], r[0] - 1]) / (2 * self.dx)
        dz_dy = (self.curr_field[r[1] + 1, r[0]] - self.curr_field[r[1] - 1, r[0]]) / (2 * self.dy)
    
        return np.array([dz_dx, dz_dy]) / np.linalg.norm([dz_dx, dz_dy])

    def get_state_vector(self, r):
        # state vector = [r_x, r_y, z_r, z_grad_x, z_grad_y, z_dot]
        state_vector = []
        # adding r_x, r_y
        state_vector.append(r[0])
        state_vector.append(r[1])

        # adding the field value at r_x, r_y
        state_vector.append(self.curr_field[r[1], r[0]])
        # adding the gradient of the field at r_x and r_y
        z_grad = self.get_gradient(r)
        state_vector.append(z_grad[0])
        state_vector.append(z_grad[1])

        return state_vector

    def show_field_state(self, num):
        # self.fig_field_ax.cla()
        # im = self.fig_field_ax.imshow(self.curr_field, cmap="Blues")
        fig = plt.figure(figsize=(8, 8))
        fig_ax1 = fig.add_subplot(121)
        fig_ax1.set_title('Field State')
        fig_ax1.set_aspect('equal')

        fig_ax1.imshow(self.curr_field, cmap="Blues", origin=(100, 0))

        # Plot trajectory
        traj_r = [p[0] for p in self.trajectory]
        traj_c = [p[1] for p in self.trajectory]
        fig_ax1.plot(traj_r, traj_c, 'o', color='black')

        # Plot starting position
        fig_ax1.plot(self.init_position[0], self.init_position[1], '*')

        # Plot ending position
        fig_ax1.plot(self.dest_position[0], self.dest_position[1], '*', color='red')
 
        # self.fig_field.colorbar(im, orientation='vertical')
        # plt.show()

        fig_ax2 = fig.add_subplot(122)
        fig_ax2.set_title('Agent Field State')
        fig_ax2.set_aspect('equal')

        fig_ax2.imshow(self.agent_field_state, cmap="Blues", origin=(100, 0))
        
        # Plot starting position
        fig_ax2.plot(self.init_position[0], self.init_position[1], '*')

        # Plot ending position
        fig_ax2.plot(self.dest_position[0], self.dest_position[1], '*', color='red')

        path = "./images/"
        plt.savefig(path + str(num) + ".png")
        print("<<<<<<<<<<<<<<<<<<<<< Iter:" + str(num) + " Image Saved! >>>>>>>>>>>>>>>>>>>>>>>")
        plt.close(fig=fig)

        # TODO(deepak): Add title, time instant etc
