import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
#forward-pass
def forward_pass(X,weight1,bias1,weight2,bias2):
    z1 = np.dot(weight1,X) + bias1
    a1 = sigmoid(z1)
    z2 = np.dot(weight2, a1) + bias2
    output = sigmoid(z2)
    return z1,a1,z2,output

def backward_propagation(X,y,output,z2,weight2,a1,z1,weight1,bias1,bias2,learning_rate):
    """
    w_new = w - η * dL/dw
    chain_rule: dL/dw = dL/dy_hat * dy_hat/dz2 * dz2/da1 * da1/dz1 * dz1/dw
    """

    m = X.shape[1]
    #output layer
    dL_doutput = 2 * (output-y) / m #mse derivative
    doutput_dz2 = derivative_sigmoid(z2)

    delta2 = dL_doutput * doutput_dz2

    #hidden layer
    dL_da1 = np.dot(weight2.T,delta2)
    da1_dz1 = derivative_sigmoid(z1)

    delta1 = dL_da1 * da1_dz1

    #gradients
    dW2 = np.dot(delta2, a1.T)
    db2 = np.sum(delta2, axis=1, keepdims=True)

    dW1 = np.dot(delta1,X.T)
    db1 = np.sum(delta1, axis=1, keepdims=True)

    #update

    weight2 -=learning_rate * dW2
    bias2 -= learning_rate * db2

    weight1 -= learning_rate * dW1
    bias1 -= learning_rate * db1

    return weight1, bias1, weight2, bias2

#inputs and outputs
iris_df = pd.read_csv('Iris.csv')
#print(iris_ds)

X = iris_df.drop(columns=["Id", "Species"]).values

y = iris_df["Species"]

y_encoded = pd.get_dummies(y).values


"""sns.pairplot(iris_df, hue="Species")
plt.show()
iris_df.head()"""

X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,test_size = 0.2,random_state=42)

X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

np.random.seed(42)


#layer1
weight1 = np.random.randn(4,4)
bias1 = np.random.randn(4,1)

#layer2
weight2 = np.random.randn(3,4)
bias2 = np.random.randn(3,1)
print("X:", X_train.shape)
print("y:", y_train.shape)
print("W1:", weight1.shape)
print("W2:", weight2.shape)



epochs = 2000
for i in range(epochs):

    z1, a1, z2, output = forward_pass(
        X_train, weight1, bias1, weight2, bias2
    )
    z1_test, a1_test, z2_test, output_test = forward_pass(
        X_test, weight1, bias1, weight2, bias2
    )

    weight1, bias1, weight2, bias2 = backward_propagation(
        X_train,
        y_train,
        output,
        z2,
        weight2,
        a1,
        z1,
        weight1,
        bias1,
        bias2,
        learning_rate=0.1
    )


    if i % 200 == 0:
        loss = mse(y_train, output)
        print(f"Epoch {i}, Loss: {loss}")
        predictions = np.argmax(output, axis=0)
        true_labels = np.argmax(y_train, axis=0)
        pred_test = np.argmax(output_test, axis=0)
        true_test = np.argmax(y_test, axis=0)
        print("train accuracy score:",accuracy_score(true_labels,predictions))
        print("test accuracy score:", accuracy_score(true_test, pred_test))



X_vis = X_train[2:4, :]
y_vis = np.argmax(y_train, axis=0)

# grid oluştur
x_min, x_max = X_vis[0].min()-1, X_vis[0].max()+1
y_min, y_max = X_vis[1].min()-1, X_vis[1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()].T

# boyut 4 olduğu için diğer 2 feature’ı 0 koyuyoruz
zeros = np.zeros((2, grid.shape[1]))
grid_full = np.vstack((grid, zeros))

_, _, _, output_grid = forward_pass(
    grid_full, weight1, bias1, weight2, bias2
)

Z = np.argmax(output_grid, axis=0)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_vis[0], X_vis[1], c=y_vis)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary")
plt.show()
