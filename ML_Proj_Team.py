import numpy as np




#JOY - FLUSH THE MATH OUT WRITE DETAILED EQUATIONS HERE
#     W is the weight
#     b is the bias
#     A0 is input layer
#     Z1 is the unactivated firstlayeer, we apply a Weight W1 and input A1 and a Bias B1 tro it
#     Therefore Z1 = W1 * A0 + B1
#     Z1 goes into relu outcomes A1 which goes to Z2 = W2*A2 + b2
#     Since A2 is gonna go to the output later we pass Z2 to softmax and we get A2 out
#    



def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((2, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2
