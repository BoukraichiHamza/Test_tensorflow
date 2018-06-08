import matplotlib.pyplot as plt
import numpy as np
"""
	Predict lambda in d/dx(y) = lambda*y
"""
"""
	Global Variables
"""

n = 5 #number of nodes in first layer 

def activation(z):
    """
        Activation/Sigmoid
    """
    return 1 / (1 + np.exp(-z))


def derivative_activation(z):
    """
        Derivative of the activation/Sigmoid

    """
    return activation(z) * (1 - activation(z))


def pre_activation(features, w,v,b,d):
    """
        Compute the pre activation
        **input: **
            *features
            *w : first layer weights
            *v : output node weight
            *b : bias of first layer
            *d : output node bias        
    """
    y = np.dot(features, w) + b
    return np.dot(features, v) + d


def init_variables():
    """
        Init model variables (weights and bias)
    """
    w = np.random.randn(n,n)
    print('w =',w)
    v = np.random.randn(n,1)
    print('v =',v)
    b = np.zeros([n,1])
    print('b =',b)
    d = 0
    print('d =',d)

    return w,v,b,d


def predict(features,w,v,b,d):
    """
        Predict the targets
        **input: **
            *features
            *w : first layer weights
            *v : output node weight
            *b : bias of first layer
            *d : output node bias    
        **return: predicted targets 
    """
    z1 = pre_activation(features, w, b)
    y1 = activation(z1)
    z = pre_activation(z1, v, d)
    y = activation(z)
    return y


def get_dataset():
    """
        Method used to generate the dataset : TO DO
    """


def cost(predictions, targets):
    """
        Compute the cost of the model
        **input: **
            *predictions: (Numpy vector) y
            *targets: (Numpy vector) t
    """
    return np.mean((predictions - targets) ** 2)


def train(features, targets, w,v,b,d):
    """
        Method used to train the model using Levenberg Marquadt Method
        **input: **
            *features
            *targets
            *w : first layer weights
            *v : output node weight
            *b : bias of first layer
            *d : output node bias
        **return dw,dv,db,dd*
            *update first layer weights
            *update output node weight
            *update first layer bias
            *update output node weight
    """
    epochs = 100
    learning_rate = 0.1

    # Print current Accuracy
    predictions = predict(features, weights, bias)
    print("Accuracy = %s" % np.mean(predictions == targets))


    for epoch in range(epochs):
        # Compute and display the cost every 10 epoch
        if epoch % 10 == 0:
            predictions = activation(pre_activation(features, weights, bias))
            print("Current cost = %s" % cost(predictions, targets))
        
        # Appel Levenberg Marquadt
        

    # Print current Accuracy
    predictions = predict(features, weights, bias)
    print("Accuracy = %s" % np.mean(predictions == targets))


if __name__ == '__main__':
	init_variables()
	"""
    # Dataset
    features, targets = get_dataset()
    # Variables
    weights, bias = init_variables()
    # Train the model
train(features, targets, weights, bias)
"""
