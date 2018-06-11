import numpy as np

def generate_data(m,n):
    train_data   = []
    train_target = []
    test_data    = []
    test_target  = []
    for i in range(1,n+1):
        new_data1 = []
        new_data2 = []
        lambda1   = np.random.randn()
        lambda2   = np.random.randn()

        for j in range(-m,m):
            new_data1.append(np.exp(-lambda1*j/m))
            new_data2.append(np.exp(-lambda2*j/m))

        train_data.append(new_data1)
        test_data.append(new_data2)
        train_target.append([lambda1])
        test_target.append([lambda2])

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
