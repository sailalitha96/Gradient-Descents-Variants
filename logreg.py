'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''


import numpy as np 
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
from numpy import linalg as LA

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, regNorm=2, epsilon=0.0001, maxNumIters = 5000):
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
        self.theta_final=   None
        self.hist= None 
        

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
            cost = (-y.T * np.log(self.sigmoid(X*theta)) - (1.0-y).T* np.log( 1.0 - self.sigmoid(X*theta))) + (self.regLambda*(theta.T * theta)/ 2.0)  
        
        elif self.regNorm == 1 :
            cost = (-y.T * np.log(self.sigmoid(X*theta)) - (1.0-y).T* np.log( 1.0 - self.sigmoid(X*theta))) + (self.regLambda*(np.sum(theta))/ 2.0) 
            
        c= cost.item((0,0))
#        print(c)
#        
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
        X = np.array(X)
        n,d = X.shape
#        print("n and d , yhat and then theta, regamatrix*thetA")
#        print(X)
#        print(y)
#        print(theta)
        
        yhat = self.sigmoid(X* theta)
#        print(yhat.shape)
       
     
        
#        yhat = self.sigmoid(X * theta)
#        regmatrix =  regLambda * np.ones((d,1))
#
#        regmatrix[0]= 0 
        
       

#        gradient = (X.T* (yhat - y) ) + ((regmatrix).T * theta) 
#        gradient= gradient.reshape((d,1)) # doubt
        #gradient= np.asarray(gradient)
        
        if self.regNorm == 2 :
             gradient = (X.T * (yhat - y) ) + (regLambda*theta)
        else :
             gradient = (X.T* (yhat - y) ) + (regLambda *  (np.sign(np.array(theta))))  #(np.array(theta))
            
        
        gradient[0] = sum(self.sigmoid(X*theta)- y )
        
#        print(gradient.shape)
        return gradient 

        

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        
        n,d = X.shape
        X = np.c_[np.ones((n,1)), X]
#        init_theta= np.matrix(np.random.randn((d+1))).T
        self.theta=  np.matrix(np.random.randn((d+1))).T
        print(self.theta.shape)
        theta_updated = self.theta
        theta_old = self.theta
        j=0
        jp=[]
        self.hist=[]
        while j < self.maxNumIters:
            
            theta_updated = theta_old - (self.alpha* (self.computeGradient(self.theta, X, y, self.regLambda)))
           
            if (self.hasConverged(theta_old,theta_updated)):
                self.theta = theta_updated 
                
                break
            
            else :
                theta_old = theta_updated
                j=j+1
                print(j)
                jp.append(j)
                cost = self.computeCost(theta_updated, X, y, self.regLambda)
                self.hist.append(cost)
#                print(cost)
                self.theta = theta_updated
                
       
    
        
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
        
#        print(y_pred)
        for i in range(y_pred.shape[0]):
            if y_pred[i] > 0.5:
                y_pred[i] = 1
            else :
                y_pred[i]=0 
#                
        return np.array(y_pred)


    def sigmoid(self, Z):
        
        return 1.0/(1.0+np.exp(-Z))
    
        
    def hasConverged(self,theta_old,theta_updated):
        '''
        checks for a n epsilon value
        '''
        n = theta_updated - theta_old
        if LA.norm(n) <= self.epsilon:
            return 1 
        else :
            return 0
            
            