import matplotlib.pyplot as plt
import numpy as np
import time

# Create an initial array of angles
theta = np.linspace(0, 2 * np.pi, 100)

# Create the initial radii array
r = 1 + 0.5 * np.sin(4 * theta)

# Create the figure and polar subplot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
line, = ax.plot(theta, r)

# Annotation and arrow placeholders
z_label = ax.annotate('z', xy=(0, 1), xytext=(0, 1.1), textcoords='data', ha='center', va='center', fontsize=12)
y_label = ax.annotate('y', xy=(np.pi/2, 1), xytext=(np.pi/2, 1.1), textcoords='data', ha='center', va='center', fontsize=12)
arrow = ax.annotate('', xy=(0, 1.5), xytext=(0, 0), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))

# Update the plot in a loop
for i in range(10):
    # Update the data
    r = 1 + 0.5 * np.sin(4 * theta + i * np.pi / 10)
    line.set_ydata(r)
    
    # Update annotations (if necessary)
    z_label.set_position((0, 1.1))
    y_label.set_position((np.pi/2, 1.1))
    arrow.xy = (0, 1.5)
    arrow.xytext = (0, 0)
    
    # Redraw the plot
    plt.draw()
    
    # Pause for 1 second
    plt.pause(1)

plt.show()