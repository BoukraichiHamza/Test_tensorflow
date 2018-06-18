import numpy as np

"""
    Global Variables
"""
LAMBDA = 1
alpha =  3
beta = -2
gamma =  1

"""
Options :
 1 = exp(lambda*x)
 2 = ax+b
 3 = ax^2+bx+c
"""

def generate_data(n,option):
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
		if (option == 1):
			f1 = np.exp(LAMBDA*j1)
			f2 = np.exp(LAMBDA*j2)
		elif (option == 2):
			f1 = alpha*j1+beta
			f2 = alpha*j2+beta
		else:
			f1 = alpha*j1**2+beta*j1+gamma
			f2 = alpha*j2**2+beta*j2+gamma
			
		train_target.append([f1])
		test_target.append([f2])
	
	return train_data,train_target,test_data,test_target

def error_f(x,target):
    return np.square(np.exp(LAMBDA*x)-target)

if __name__ == '__main__':
    """
    Test
    """
    td,tt,td1,tt1 = generate_data(5,3)
    print('Train_data = ',td)
    print('Train_target = ',tt)
    print('Test_data = ',td1)
    print('Test_target = ',tt1)

    print(error_f(10,100))
