'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np 
import math
import random
from numpy import linalg as LA
import matplotlib.pyplot as plt



class LogisticRegressionAdagrad:

    def __init__(self, alpha = 0.01, regLambda=0.01, regNorm=2, epsilon=0.01, maxNumIters = 10000):
        '''
        Constructor
        Arguments:
        	alpha is the learning rate
        	regLambda is the regularization parameter
        	regNorm is the type of regularization (either L1 or L2, denoted by a 1 or a 2)
        	epsilon is the convergence parameter
        	maxNumIters is the maximum number of iterations to run
        '''

        self.alpha= alpha
        self.regLambda = regLambda
        self.regNorm = regNorm 
        self.epsilon = epsilon 
        self.maxNumIters = maxNumIters
        self.theta = None
        self.eta = 0.00001

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        n,d = X.shape
        if self.regNorm == 2 :
            cost = (-y.T * np.log(self.sigmoid(X*theta)) - (1.0-y).T* np.log( 1.0 - self.sigmoid(X*theta))) + (regLambda*(theta.T * theta)/ 2.0)  
        
        elif self.regNorm == 1 :
            cost = (-y.T * np.log(self.sigmoid(X*theta)) - (1.0-y).T* np.log( 1.0 - self.sigmoid(X*theta))) + (regLambda*(np.sum(theta))/ 2.0) 
            
        c= cost.item((0,0))
#        print(c)
        
        return c

        
    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        X = np.asarray(X)
        n,d = X.shape

        yhat = self.sigmoid(np.matmul(X,theta))
#        regmatrix =  regLambda* np.ones((d,1))
#        
#        
#        regmatrix[0,0]= 0 
#       
#        gradient = (X.T * (yhat - y) ) + (regLambda*theta)
#        gradient[0] = sum(self.sigmoid(X*theta)- y )
#        gradient = (X.T* (yhat - y) ) + ((regmatrix).T*theta) 
        
        if self.regNorm == 2 :
              gradient = (X.T * (yhat - y) ) + (regLambda*theta)
        else :
            gradient = (X.T* (yhat - y) ) + (regLambda *  (np.sign(np.array(theta))))  #(np.array(theta))
        
    
        gradient[0] = sum(self.sigmoid(X*theta)- y )
    
#        print(gradient.shape)
        return gradient 


       
        return gradient 

        

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n,d = X.shape
#        print(X.shape)
        X = np.c_[np.ones((n,1)), X]
#        init_theta= np.matrix(np.random.randn((d+1))).T
        self.theta=  np.matrix(np.random.randn((d+1))).T
        theta = self.theta
        print(self.theta.shape)
        j=0
#        gradient = np.zeros((d+1,1))
        total_grad = np.zeros((d+1,1))
     
      
        for j in range(self.maxNumIters):
            
            
            idx= np.random.randint(0,X.shape[0])
            test_X= X[idx,:].reshape(1,d+1)
            test_y= y[idx,:]
            
            # uncommnet two lines below for sgd 
            
#            theta= theta- self.alpha*( self.computeGradient(self.theta, test_X, test_y,self.regLambda))
#            self.theta= theta
#            
            ## uncomment below for sgd- adagrad
            
            gradient = self.computeGradient(self.theta, test_X, test_y,self.regLambda)
            
            total_grad = total_grad + np.square(gradient)
            
            for i in range(len(gradient)):
                self.theta = self.theta - (self.alpha * gradient /(np.sqrt(total_grad[i]) +self.eta))
                cost = self.computeCost(self.theta, X, y, self.regLambda)
            
#            print(gradient)
            
            
                
                
                
    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''

        n,d= X.shape
        print(self.theta)
        X = np.c_[np.ones((n,1)), X]
#        y_pred = self.sigmoid(X.dot(self.theta))
        y_pred=self.sigmoid(np.matmul(X,self.theta))
#        print(X*self.theta)
        
        for i in range(y_pred.shape[0]):
            if y_pred[i] >= 0.5:
                y_pred[i] = 1
            else :
                y_pred[i]=0 
            
                
        return np.array(y_pred)


    def sigmoid(self, Z):
    	
        return 1/(1+np.exp(-Z))
    
    def hasConverged(self,theta_old,theta_updated):
        '''
        checks for a n epsilon value
        '''
        n = theta_updated - theta_old
        if LA.norm( n, self.regNorm) <= self.epsilon:
            return 1 
        else :
            return 0