import numpy as np

def LM(x,y,F,J,tol,maxit,LAMBDA) :
	
	"""
	 Levenberg-Marquardt method for large scale nonlinear least squares
	 problems 
	 Input: x     : variable with respect to which we solve for (weights)
	        y     : training sample
	        F     : function (-nabla ANN(u(y,x))-f(x))
	        J     : function handle that computes Jx and J'x
	        tol   : stopping tolerance 
	        maxit : max number of iteration
	        
	"""
	
	"""
		Initialisation
	"""
	
	# maximum, minimum  of LAMBDA
	LAMBDA_max = 1.e+6
	LAMBDA_min = 1.e-4
	LAMBDAk    = LAMBDA
	
	#Iterations counter
	nbiter = 0
	
	#Parameters
	eta    = 0.1
	gamma1 = 0.85
	gamma2 = 1.5
	
	#Iterate
	xk = x
	
	#Function, Jacobian and gradient evaluation
	fx   = F(xk,y)
	jac  = J(xk,y)
	grad =  np.matmul(np.transpose(jac),fx)

	
	#Actual norm of gradient
	normgrad = np.linalg.norm(grad)
	
	""" 
		Computation
	"""
	while (normgrad > tol) & (nbiter < maxit):
		#inc iter
		nbiter += 1
		
		#Compute the step pk
		n		 = np.shape(jac)[1]
		jacjac    = np.matmul(np.transpose(jac),jac)
		jac_reg   = np.add(jacjac,np.dot(np.eye(n),LAMBDAk))
		pk        = np.linalg.solve(jac_reg,np.dot(grad,-1))
		
		#Compute rhok
		xkaux = np.add(xk,pk)
		ared  = np.linalg.norm(fx)**2 - np.linalg.norm(F(xkaux,y))**2
		pred  = np.linalg.norm(fx)**2
		pred  = pred - np.linalg.norm(np.add(fx,np.matmul(jac,pk)))**2
		rhok  = ared/pred
		
		#update iterate
		if (rhok > eta):
			xk      = xkaux
			fx      = F(xk,y)
			jac     = J(xk,y)
			grad    = np.matmul(np.transpose(jac),fx)
			LAMBDAk = gamma1*LAMBDAk
			normgrad = np.linalg.norm(grad)
		else:
			LAMBDAk = gamma2*LAMBDAk
	return xk,nbiter
		

if __name__=="__main__":
	"""
		Tests
	"""
	
	A = np.array([[2,0],[0,4]])
	x = np.array([0.2,0.2])
	y = np.array([1,1])
	
	def Ftest(x,y):
		Ax = np.matmul(A,x)
		return np.add(Ax,np.dot(y,-1))
		
	def Jtest(x,y):
		return A
	
	x,iters = LM(x,y,Ftest,Jtest,1e-6,1000,0.05)
	print('nbiter = ',iters)
	print('sol    = ',x)
	
	
	
	


 
