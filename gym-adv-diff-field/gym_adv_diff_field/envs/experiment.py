import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy import io
import os

class Experiment:

    def __init__(
        self,
        field_size=[100, 100],
        field_vel=[-0.4, 0.2],
        grid_size=[0.8, 0.8],
        init_position=[10, 10],
        dest_position=[90, 90]):

        self.field_size = field_size
        self.field_vel = field_vel
        self.dx = grid_size[0]
        self.dy = grid_size[1]
        self.dt = 0.1

        self.init_position = None
        self.dest_position = None

        self.agent_field_state = np.zeros(field_size)

        self.curr_field = self.create_field()
        self.prev_field = np.zeros(field_size)

        self.trajectory = []

    def set_init_position(self, init_position):
        if (init_position[0] < 0 or init_position[0] > self.field_size[0]) or (init_position[1] < 0 or init_position[1] > self.field_size[1]):
            print("[ERROR] Initial position out of range")
        self.init_position = init_position
    
    def set_dest_position(self, dest_position):
        if (dest_position[0] < 0 or dest_position[0] > self.field_size[0]) or (dest_position[1] < 0 or dest_position[1] > self.field_size[1]):
            print("[ERROR] Destination position out of range")
        self.dest_position = dest_position
    
    def create_field(self):
        cwd = os.getcwd()
        loaded_mat = io.loadmat(cwd + "/u.mat")
        u = loaded_mat.get('u')

        u_1 = u.copy()
        source_1_c_shift = 13

        for r in range(0, self.field_size[0]):
            for c in range(0, self.field_size[1]):
                u_1[r, c] = u[r % self.field_size[0], (c - source_1_c_shift) % self.field_size[1]]

        reverse_u = u.copy()

        for r in range(0, self.field_size[0]):
            for c in range(0, self.field_size[1]):
                reverse_u[r, c] = u[self.field_size[0] - r - 1, self.field_size[1] - c - 1]

        reverse_u_1 = u.copy()
        source_2_r_shift = 10
        source_2_c_shift = 2

        for r in range(0, self.field_size[0]):
            for c in range(0, self.field_size[1]):
                reverse_u_1[r, c] = reverse_u[(r + source_2_r_shift) % self.field_size[0], (c + source_2_c_shift) % self.field_size[1]]

        combined_field = u_1 + reverse_u_1
        return combined_field

    def show_curr_field(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_title('Visualizing combined field')
        ax.set_aspect('equal')

        plt.imshow(self.curr_field, cmap='Blues')
        plt.colorbar(orientation='vertical')
        plt.show()

