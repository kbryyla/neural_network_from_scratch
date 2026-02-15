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

def derivative_sigmoid(x):
    return sigmoid(x) * ( 1-sigmoid(x))

def mse(y, y_hat):
    return np.mean((y-y_hat) ** 2)

def backword_propagation(X,y,output,z2,weight2,a1,z1,weight1,learning_rate):
    """
    w_new = w - η * dL/dw
    chain_rule: dL/dw = dL/dy_hat * dy_hat/dz2 * dz2/da1 * da1/dz1 * dz1/dw
    """
    dL_doutput = 2 * (y-output) #mse derivative
    doutput_dz2 = derivative_sigmoid(z2)

    delta2 = dL_doutput * doutput_dz2 # weight * input

    dL_da1 = np.dot(weight2.T,delta2)
    da1_dz1 = derivative_sigmoid(z1)

    delta1 = dL_da1 * da1_dz1

    #gradients
    dW2 = np.dot(delta2, a1.T)
    db2 = delta2

    dW1 = np.dot(delta1,X.T)
    db1 = delta1

    #update

    weight2 -=learning_rate * dW2
    bias2 -= learning_rate * db2

    weight1 -= learning_rate * dW1
    bias1 -= learning_rate * db1

    return weight1, bias1, weight2, bias2


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


#forward-pass
z1 = np.dot(weight1,X) + bias1
a1 = sigmoid(z1)
z2 = np.dot(weight2, a1) + bias2
output = sigmoid(z2)
loss = mse(y, output)

print(weight1.shape)
print(X.shape)
print(z1.shape)
print(z2.shape)

print("output:", output)
print("loss:", loss)