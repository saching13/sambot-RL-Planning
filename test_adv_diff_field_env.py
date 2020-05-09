import gym
import gym_adv_diff_field

import pickle
import numpy as np
import matplotlib.pyplot as plt


file = np.load("curr_field.npy")
print(file.shape)
sliced_size = file[::2, ::2]
print(sliced_size.shape)

fig = plt.figure(figsize=(8, 8))
fig_ax1 = fig.add_subplot(111)
fig_ax1.set_title('Field State')
fig_ax1.set_aspect('equal')

fig_ax1.imshow(sliced_size, cmap="Blues", origin=(100, 0))

# fig_ax2 = fig.add_subplot(122)
# fig_ax2.set_title('Agent Field State')
# fig_ax2.set_aspect('equal')
#
# fig_ax2.imshow(sliced_size, cmap="Blues", origin=(100, 0))

# Plot starting position
# fig_ax2.plot(self.init_position[0], self.init_position[1], '*')

# Plot ending position
# fig_ax2.plot(self.dest_position[0], self.dest_position[1], '*', color='red')

path = "./"
plt.savefig(path + "static" + ".png")
# print("<<<<<<<<<<<<<<<<<<<<< Iter:" + str(num) + " Image Saved! >>>>>>>>>>>>>>>>>>>>>>>")
plt.close(fig=fig)