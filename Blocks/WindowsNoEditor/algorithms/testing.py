import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def xyz(args, real_t):
  """
  This function generates 3D coordinates based on given arguments and real_t values.

  Args:
      args: A tuple containing period, size of x, and size of y (all integers).
      real_t: A numpy array of real values.

  Returns:
      A tuple containing 3 numpy arrays representing x, y, and z coordinates.
  """
  period, sizex, sizey = args
  if not period:
    period = real_t[-1]
  t = real_t / period * 2 * np.pi
  x = np.sqrt(2) * np.cos(t) / (1 + np.sin(t) ** 2)
  y = x * np.sin(t)
  x = sizex * x
  y = sizey * y
  z = np.ones_like(x) * 1.5
  return x, y, z

# Call the function with your desired arguments
period = 10
sizex = 1
sizey = 1
count = 10  # Number of points for the real_t array
real_t = np.linspace(0, period, count)  # Create equally spaced real_t values

# Get the x, y, and z coordinates from the function
x, y, z = xyz([period, sizex, sizey], real_t)

# Create the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the points with a surface plot
ax.plot_trisurf(x, y, z, cmap='viridis', alpha=0.8)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot from xyz function')

plt.show()