import matplotlib.pyplot as plt
import numpy as np
"""
	Predict lambda in d/dx(y) = lambda*y
"""
"""
	Global Variables
"""

n = 5 #number of nodes in first layer

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
