import numpy as np
import matplotlib.pyplot as plt

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f'x_train: {x_train}')
print(f'y_train: {y_train}')

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

# calculate m using the len() function
m = len(x_train)
print(f"Number of training examples is: {m}")

# Training Example
i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f'(x^({i}), y^({i})) = {x_i}, {y_i}')

# Model Parameters w, b
w = 200
b = 100
print(f'w: {w}')
print(f'b: {b}')

# Model Function
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0] 
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    
    return f_wb


# Compute the model output
tmp_f_wb = compute_model_output(x_train, w, b)


# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Model Prediction')
# Plot the training data
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
# Set the title
plt.title('Housing Prices')
# Set the x-axis and y-axis labels
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.legend()
plt.show()

# Prediction

x_i = 1.5
cost_sqft = w * x_i + b
print(f'Cost of the house is: {cost_sqft:.0f}')

# Cost Function
def compute_cost(x , y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
    total_cost (float): The cost of using w,b as the parameters for 
    linear regression to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b           # a prediction is calculated
        cost = (f_wb - y[i]) ** 2     # the squared difference between the target value and the prediction.
        cost_sum += cost              # these differences are summed over all the examples and divided by 2m to produce the total cost
    total_cost = (1 / (2 * m)) * cost_sum
    
    return total_cost


J_wb = compute_cost(x_train, y_train, w, b)
print(f'J(w, b) = {J_wb}')

