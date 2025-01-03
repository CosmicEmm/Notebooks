import numpy as np
import matplotlib.pyplot as plt


x_train = np.array([35, 54, 55, 55, 47, 54, 48, 46, 44])
y_train = np.array([33, 53, 60, 55, 51, 61, 51, 42, 44])

m = x_train.shape[0]
print(m)

w = 1
b = 1

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(9):
        f_wb[i] = w * x[i] + b
    
    return f_wb

y_hat = compute_model_output(x_train, w, b)
print(y_hat)

# Plotting Predictions
plt.plot(x_train, y_hat, c='g', label='Model Prediction')
# Plotting Actual Targets
plt.scatter(x_train, y_train, marker='x', c='r', label='Targets')
plt.title('Goals Scored by Cristiano Ronaldo')
plt.xlabel('No. of Appearances')
plt.ylabel('Goals Scored')
plt.legend()
plt.show()

# Making Prediction
x_i = 51

f_wb = w * x_i + b
print(f'Goals scored: {f_wb}')



