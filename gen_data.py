import numpy as np

def generate_data(n,LAMBDA=1):
    train_data   = []
    train_target = []
    test_data    = []
    test_target  = []
    interval     = []

    for i in range(1,n+1):
        j1 = np.random.randn()
        j2 = np.random.randn()
        train_data.append([np.exp(LAMBDA*j1)])
        test_data.append([np.exp(LAMBDA*j2)])
        train_target.append([j1])
        test_target.append([j2])

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
