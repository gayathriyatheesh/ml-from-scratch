# Day 3 – MLP: Gradient Checking and Evaluation

## 1. Numerical Gradient Checking

- Numerical gradient checking verifies whether our backpropagation gradients are correct.
- It estimates gradients by slightly increasing and decreasing a parameter.
- We use `epsilon = 1e-5` for small parameter changes.
- We compare numerical gradients with analytical gradients from backpropagation.
- If the difference is very small, our gradient calculations are likely correct.

## 2. Analytical vs Numerical Gradients

- Analytical gradient: Calculated using the chain rule in backpropagation.
- Numerical gradient: Estimated using changes in the loss.
- Both should be approximately equal.

**Result:** Successfully checked gradients for W1, W2, b1 and b2. All five tested parameter elements passed.

## 3. Testing on Unseen Data

- Training data is used to learn the weights and biases.
- Test data is used to evaluate model performance on new examples.
- We must not train on test data because it would make the evaluation unfair.
- Low training and test losses indicate that the model is learning and generalizing well.

**Results:**
- Training loss (epoch 900): 0.005626
- Test loss: 0.003971

## 4. What I Built

Implemented a two-layer neural network (MLP) from scratch using NumPy:

- Parameter initialization
- Forward propagation
- Mean squared error loss
- Backpropagation
- Gradient descent
- Numerical gradient checking
- Evaluation on unseen data

**Day 3 status: Completed.**