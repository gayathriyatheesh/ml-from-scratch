
# Day 2 - Backpropagation Derivation

## Forward Pass

z1 = W1 * x + b1
a1 = sigmoid(z1)
z2 = W2 * a1 + b2

Loss = 1/2 * (z2 - y)^2

## Backward Pass

1. Output gradient

dL/dz2 = z2 - y

2. Second layer gradients

dL/dW2 = (z2 - y) * a1
dL/db2 = z2 - y

3. Hidden activation gradient

dL/da1 = (z2 - y) * W2

4. Hidden layer gradient

dL/dz1 = (z2 - y) * W2 * a1 * (1 - a1)

5. First layer gradients

dL/dW1 = dL/dz1 * x
dL/db1 = dL/dz1

## Key Learning

Backpropagation uses the chain rule to calculate
how much each weight and bias contributes to the loss.
Gradient descent uses these gradients to update parameters.
