import numpy as np
x = np.linspace(0,10,100)
noise = np.random.normal(loc=0, scale=0.5, size=100)
y = 3 * x + 2 + noise
w = 0.0
b = 0.0
y_pred = w * x + b
print(y_pred[:5])
def mse(y_true, y_pred):
    difference =y_pred - y_true
    squared_difference = difference ** 2
    return np.mean(squared_difference)
loss = mse(y, y_pred)
print(loss)
def compute_gradients(x, y_true, y_pred):
    errors = y_pred - y_true
    dw = 2 * np.mean(errors * x)
    db = 2 * np.mean(errors)
    return dw, db
learning_rate = 0.01

for i in range(1000):
    y_pred = w * x + b
    dw, db = compute_gradients(x, y, y_pred)
    w -= learning_rate * dw
    b -= learning_rate * db
    y_pred = w * x + b
    loss = mse(y, y_pred)
    if i % 100 == 0:
        print(f"Iteration {i:4d}: Loss = {loss:.4f}")
print(f"w = {w:.4f}, b = {b:.4f}")
