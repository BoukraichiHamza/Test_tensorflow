import numpy as np

"""
    Global Variables
"""
LAMBDA = 1
def generate_data(n):
    train_data   = []
    train_target = []
    test_data    = []
    test_target  = []
    interval     = []

    for i in range(1,n+1):
        j1 = np.random.randn()
        j2 = np.random.randn()
        train_data.append([j1])
        test_data.append([j2])
        train_target.append([np.exp(LAMBDA*j1)])
        test_target.append([np.exp(LAMBDA*j2)])

    return train_data,train_target,test_data,test_target

def error_f(x,target):
    return np.square(np.exp(LAMBDA*x)-target)

if __name__ == '__main__':
    """
    Test
    """
    td,tt,td1,tt1 = generate_data(5)
    print('Train_data = ',td)
    print('Train_target = ',tt)
    print('Test_data = ',td1)
    print('Test_target = ',tt1)

    print(error_f(10,100))
