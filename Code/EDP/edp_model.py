import numpy as np
import gen_data_edp as gde
def sigmoid(z):
    """
        Activation/Sigmoid
    """
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z):
    """
        Derivative of the activation/Sigmoid

    """
    return sigmoid(z) * (1 - sigmoid(z))


def ANN(x, v,w,b,d):
    """
        Compute the output of the ANN
        **input: **
            *x : features
            *v : first layer weights
            *w : output node weight
            *b : bias of first layer
            *d : output node bias
    """
    y = sigmoid(np.dot(v,np.transpose(x))+b)
    print(np.shape(y))
    return np.matmul(np.transpose(w), y) + d

def dx_ANN(x, v,w,b,d):
    """
        Compute the output of the derivated ANN
        **input: **
            *x : features
            *v : first layer weights
            *w : output node weight
            *b : bias of first layer
            *d : output node bias
    """
    y = np.dot(v,np.transpose(x))+b
    z = d_sigmoid(np.matmul(np.transpose(w), y) + d)
    return np.dot(np.transpose(v),w)*z


def jac_ANN(x,v,w,b,d):
    """
        Compute the output of the Jacobian(weights) of the ANN
        **input: **
            *x : features
            *v : first layer weights
            *w : output node weight
            *b : bias of first layer
            *d : output node bias
    """
    t = np.shape(x)[0]
    s = np.shape(v)[0]
    jac = np.zeros([t,3*s+1])
    print(np.shape(jac))
    for j in range(t):
        jac[j][0:s] =  np.matmul(np.transpose(w),d_sigmoid(np.matmul(v,x[j])+b))
        jac[j][s:2*s] = x[j]*jac[j][0:s]
        jac[j][2*s:3*s] = np.reshape(sigmoid(x[j]*v+b),[5])
        jac[j][3*s] = 1
    return jac

def init_variables(n):
    """
        Init model variables (weights and bias)
    """
    v = np.random.randn(n,1)
    w = np.random.randn(n,1)
    b = np.ones([n,1])
    d = 0


    return w,v,b,d



def cost(predictions, targets):
    """
        Compute the cost of the model
        **input: **
            *predictions: (Numpy vector) y
            *targets: (Numpy vector) t
    """
    return np.mean((predictions - targets) ** 2)

if __name__ == '__main__':
    v,w,b,d = init_variables(5)
    x1,y1,x2,y2 =  gde.generate_data(2,5)
    #print(sigmoid(np.array([1,1,1,1])))
    print(jac_ANN(x1,v,w,b,d))
