import numpy as np

"""
    Global Variables
"""
eta_edp = 2
def generate_data(n,m):
    train_data   = []
    train_target = []
    test_data    = []
    test_target  = []

    #Generate train data
    for i in range(1,n+1):
        j = np.random.rand()
        train_data.append([j])
        f = eta_edp*np.exp(eta_edp*j)
        train_target.append([f])
    #Generate Test data
    for i in range(1,m+1):
        j = np.random.rand()
        test_data.append([j])
        f = eta_edp*np.exp(eta_edp*j)
        test_target.append([f])

    return train_data,train_target,test_data,test_target


if __name__ == '__main__':
    """
    Test
    """
    td,tt,td1,tt1 = generate_data(5,3)
    print('Train_data = ',td)
    print('Train_target = ',tt)
    print('Test_data = ',td1)
    print('Test_target = ',tt1)
