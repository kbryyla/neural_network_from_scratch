import numpy as np

"""
input --> layer1 --> layer2 --> ouput

*not adding nonlineartiy
layer1 = input * weight1 + bias
layer2 = layer1 * weight2 + bias
output = layer2 
"""

#input
inputs = np.array([0.5])

#layer1
weight1 = np.array([1.2])
bias1 = np.array([0.2])

#layer2
weight2 = np.array([0.8])
bias2 = np.array([0.1])


#forward-pass
layer1 = np.dot(inputs, weight1) + bias1
layer2 = np.dot(layer1, weight2 ) + bias2

output = layer2
print("output:", output)