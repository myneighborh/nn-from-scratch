# Simple Neural Network with NumPy without Deep Learning Framework

This project is a **simple 3-layer neural network implemented using only NumPy**,  
**without any deep learning frameworks** like TensorFlow or PyTorch.

Only **forward propagation** is implemented to demonstrate the core concepts of neural networks.


## Code Overview

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def soft_max(x):
    x = x - np.max(x)  # Prevent overflow
    return np.exp(x) / np.sum(np.exp(x))

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = soft_max(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
```


## Network Architecture

```
Input (2,)
↓
Affine (W1, b1)
↓
Sigmoid
↓
Affine (W2, b2)
↓
Sigmoid
↓
Affine (W3, b3)
↓
Softmax
↓
Output (2,) ← Class probabilities
```


## Preventing Overflow in Softmax

### Basic Definition

The softmax function is defined as:

<img width="215" alt="image" src="https://github.com/user-attachments/assets/478e7a0c-08ae-48a0-bbd8-8a9a9972fe8d" />

However, when the values of \( x_i \) are large,  
calculating \( e^{x_i} \) can cause **numerical overflow**.


### Solution: Subtract the Maximum Value

```python
def soft_max(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
```

By subtracting the maximum value, we maintain numerical stability  
**without affecting the output probabilities**.

<br>

<img width="269" alt="image" src="https://github.com/user-attachments/assets/01f21145-40e8-4513-b266-18613928e444" /><br>
<img width="277" alt="image" src="https://github.com/user-attachments/assets/0d9e4e6e-d7bf-4506-a347-58d3905cb040" />

This trick ensures **stable computation while preserving the same probability distribution**.

