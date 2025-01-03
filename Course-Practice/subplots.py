import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2)
# 2-D array: [[<Axes: > <Axes: >]
#             [<Axes: > <Axes: >]]
ax[0, 0].plot([1, 2, 3], [4, 5, 6])  # First subplot
ax[0, 0].set_title('First Subplot')

ax[0, 1].plot([1, 2, 3], [6, 5, 4])  # Second subplot
ax[0, 1].set_title('Second Subplot')

ax[1, 0].plot([1, 2, 3], [7, 8, 9])  # Third subplot
ax[1, 0].set_title('Third Subplot')

ax[1, 1].plot([1, 2, 3], [9, 8, 7])  # Fourth subplot
ax[1, 1].set_title('Fourth Subplot')

for a in ax.flat: # ax.flat flattens the 2D array of axes into a 1D iterable
    a.set_xlabel('X-axis')
    a.set_ylabel('Y-axis')

# Adjust the space between the subplots
plt.subplots_adjust(hspace=0.5, wspace=0.3) # hspace: height space, wspace : width space
plt.show()