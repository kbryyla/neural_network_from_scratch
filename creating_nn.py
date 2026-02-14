import numpy as np

"""
input --> layer1 --> layer2 --> ouput

*not adding nonlineartiy
layer1 = input * weight1 + bias
layer2 = layer1 * weight2 + bias
output = layer2 

*add nonlinearity
layer1 = input * weight1 + bias
layer2 = sigmoid(layer1) * weight2 + bias
output = sigmoid(layer2)

"""
def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def mse(y, y_hat):
    return np.mean((y-y_hat) ** 2)

#inputs and outputs
X = np.array([[0.1],
              [0.4],
              [0.6],
              [0.9]])   # (4x1)


y = np.array([[0]])     # (1x1)



#layer1
weight1 = np.random.randn(4,4)
bias1 = np.random.randn(4,1)

#layer2
weight2 = np.random.randn(3,4)
bias2 = np.random.randn(3,1)

#output
weight3 = np.random.randn(1,3)
bias3 = np.random.randn(1,1)


#forward-pass
z1 = np.dot(weight1,X) + bias1
a1 = sigmoid(z1)
z2 = np.dot(weight2, a1) + bias2
a2 = sigmoid(z2)
z3 = np.dot(weight3, a2) + bias3
output = sigmoid(z3)
loss = mse(y, output)

print(weight1.shape)
print(X.shape)
print(z1.shape)
print(z2.shape)

print("output:", output)
print("loss:", loss)