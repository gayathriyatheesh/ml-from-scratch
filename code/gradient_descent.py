
import numpy as np

# 1. Create training data
np.random.seed(42)

X = np.linspace(-2, 2, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 0.1, (100, 1))

# 2. Initialize weights and biases
W1 = np.random.randn(1, 1)
b1 = np.zeros((1, 1))

W2 = np.random.randn(1, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.01
epochs = 2000


# 3. Sigmoid activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 4. Forward pass
def forward(X):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2

    return z1, a1, z2


# 5. Training loop
for epoch in range(epochs):

    # Forward pass
    z1, a1, predictions = forward(X)

    # Calculate loss
    loss = np.mean(0.5 * (predictions - y) ** 2)

    # Backward pass
    dz2 = (predictions - y) / len(X)

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2)

    da1 = dz2 @ W2.T
    dz1 = da1 * a1 * (1 - a1)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1)

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 200 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

print("\nTraining complete!")
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)
