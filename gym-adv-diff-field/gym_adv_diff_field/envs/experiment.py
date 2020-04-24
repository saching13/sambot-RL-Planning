import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy import io
import os
import time


class Experiment:

    def __init__(
            self,
            field_size=[100, 100],
            field_vel=[-0.4, 0.2],
            grid_size=[0.8, 0.8],
            init_position=[10, 10],
            dest_position=[90, 90],
            view_scope_size=10):

        self.field_size = field_size
        self.field_vel = field_vel
        self.dx = grid_size[0]
        self.dy = grid_size[1]
        self.dt = 0.1
        self.view_scope_size = view_scope_size

        self.init_position = init_position
        self.dest_position = dest_position

        self.agent_field_state = np.zeros(field_size)

        self.curr_field = self.create_field()
        self.prev_field = np.zeros(field_size)

        self.trajectory = [init_position]

        # Display
        self.fig_field = plt.figure(figsize=(8, 8))
        self.fig_field_ax = self.fig_field.add_subplot(111)
        self.fig_field_ax.set_title('Field State')
        self.fig_field_ax.set_aspect('equal')

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
        print("Experiment Reset")
        

    def create_field(self):
        cwd = os.getcwd()
        loaded_mat = io.loadmat(cwd + "/u.mat")
        u = loaded_mat.get('u')

        u_1 = u.copy()
        source_1_c_shift = 13

        for r in range(0, self.field_size[0]):
            for c in range(0, self.field_size[1]):
                u_1[r, c] = u[r %
                              self.field_size[0], (c - source_1_c_shift) %
                              self.field_size[1]]

        reverse_u = u.copy()

        for r in range(0, self.field_size[0]):
            for c in range(0, self.field_size[1]):
                reverse_u[r, c] = u[self.field_size[0] -
                                    r - 1, self.field_size[1] - c - 1]

        reverse_u_1 = u.copy()
        source_2_r_shift = 10
        source_2_c_shift = 2

        for r in range(0, self.field_size[0]):
            for c in range(0, self.field_size[1]):
                reverse_u_1[r, c] = reverse_u[(r +
                                               source_2_r_shift) %
                                              self.field_size[0], (c +
                                                                   source_2_c_shift) %
                                              self.field_size[1]]

        combined_field = u_1 + reverse_u_1
        return combined_field

    def update_field(self):
        u = self.curr_field.copy()
        updated_u = self.curr_field.copy()
        u_k = self.curr_field.copy()

        dx = self.dx
        dy = self.dy
        dt = self.dt
        vx = self.field_vel[0]
        vy = self.field_vel[1]

        for i in range(1, self.field_size[0] - 1):
            for j in range(1, self.field_size[1] - 1):
                k = 1
                updated_u[j,
                          i] = u[j,
                                 i] + k * (dt / dx**2) * ((u_k[j + 1,
                                                               i] + u_k[j - 1,
                                                                        i] + u_k[j,
                                                                                 i + 1] + u_k[j,
                                                                                              i - 1] - 4 * u_k[j,
                                                                                                               i])) + vx * (dt / dx) * ((u_k[j + 1,
                                                                                                                                             i] - u_k[j,
                                                                                                                                                      i])) + vy * (dt / dy) * (u_k[j,
                                                                                                                                                                                   i + 1] - u_k[j,
                                                                                                                                                                                                i])

        self.prev_field = self.curr_field
        self.curr_field = updated_u

    def show_field_in_loop(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_title('Visualizing combined field')
        ax.set_aspect('equal')

        for i in range(100):
            # start_time = time.time()
            ax.cla()
            im = ax.imshow(self.curr_field, cmap="Blues")
            # fig.colorbar(im, orientation='vertical')
            self.update_field()
            plt.pause(0.05)
            # dur = time.time() - start_time
            # print("Time taken: " + str(dur))
        plt.show()

    def get_z_dot(self, r):
        z_k1 = self.curr_field[r[0], r[1]]
        z_k = self.prev_field[r[0], r[1]]

        return (z_k1 - z_k) / self.dt

    def get_gradient(self, r):
        # TODO(sachin): Add gradient computation here
        return [-1, -1]

    def get_state_vector(self, r):
        state_vector = []
        state_vector.append(r[0])
        state_vector.append(r[1])

        state_vector.append(self.curr_field[r[0], r[1]])

        z_dot = self.get_z_dot(r)
        state_vector.append(z_dot)

        z_grad = self.get_gradient(r)
        state_vector.append(z_grad[0])
        state_vector.append(z_grad[1])

        return state_vector
    
    def show_field_state(self):
        self.fig_field_ax.cla()
        im = self.fig_field_ax.imshow(self.curr_field, cmap="Blues")

        # Plot trajectory
        traj_r = [p[0] for p in self.trajectory]
        traj_c = [p[1] for p in self.trajectory]
        plt.plot(traj_r, traj_c, 'o', color='black')

        # Plot starting position
        plt.plot(self.init_position[0], self.init_position[1], '*')

        # Plot ending position
        plt.plot(self.dest_position[0], self.dest_position[1], '*', color='red')

        self.fig_field.colorbar(im, orientation='vertical')
        plt.show()

        # TODO(deepak): Add title, time instant etc