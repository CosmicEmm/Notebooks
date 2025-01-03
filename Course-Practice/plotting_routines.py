import numpy as np
import matplotlib.pyplot as plt

def plt_house_x(X, y, f_wb=None, ax=None):
    ''' plot house with axis '''
    if not ax:
        fig, ax = plt.subplots(1,1)  # If ax is not provided, create a new figure and Axes

    ax.scatter(X, y, marker='x', c='r', label="Actual Value")  # Plot data points (x, y)

    ax.set_title("Housing Prices")  # Set the title of the plot
    ax.set_ylabel('Price (in 1000s of dollars)')  # Set the y-axis label
    ax.set_xlabel(f'Size (1000 sqft)')  # Set the x-axis label
    if f_wb is not None:
        ax.plot(X, f_wb, c='blue', label="Our Prediction")  # Plot prediction line if f_wb is provided
    ax.legend()  # Add a legend to the plot

