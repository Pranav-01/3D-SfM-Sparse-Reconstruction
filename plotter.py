import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# list_of_all_3d_points = np.load("all_3d_points.npy")
list_of_all_3d_points = np.load("opt_all_3d_points.npy")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for point in list_of_all_3d_points:
    x, y, z = point
    ax.scatter(x, y, z, s=1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title(f"All 3D points")

plt.show()