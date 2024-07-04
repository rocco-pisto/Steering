import matplotlib.pyplot as plt
import numpy as np

# Generate some example data (replace with your actual data)
theta = np.linspace(0, 2 * np.pi, 100)
line1_values = np.sin(theta)  # Line 1 data
line2_values = np.cos(theta)  # Line 2 data

# Create a figure with two polar subplots
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})

# Define a colormap for the gradient
cmap = plt.get_cmap('viridis')

# Plot Line 1 with gradient color
ax1.plot(theta, line1_values, color=cmap(0.0), label='Line 1')
ax1.set_title('Line 1')
ax1.set_rticks([])  # Hide radial ticks for cleaner appearance

# Plot Line 2 with the same gradient color
ax2.plot(theta, line2_values, color=cmap(1.0), label='Line 2')
ax2.set_title('Line 2')
ax2.set_rticks([])  # Hide radial ticks

# Add a colorbar to show the gradient
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
sm.set_array([])  # Dummy array for colorbar
cbar = plt.colorbar(sm, ax=[ax1, ax2], location='bottom')
cbar.set_label('Gradient')

# Customize the legend
for ax in [ax1, ax2]:
    ax.legend(loc='upper right')

# Set the overall title
plt.suptitle('Gradient-Colored Lines on Polar Subplots')
plt.tight_layout()
plt.show()
