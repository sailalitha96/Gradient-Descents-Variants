'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np
#from linreg import LinearRegression
import random

#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        #TODO
        self.degree= degree
        self.regLambda = regLambda
        self.mean= None 
        self.std = None
        self.theta = None

    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        #TODO
        X = np.array(X)
        X_array= np.zeros((X.shape[0],degree))
        row_size = X.shape[0]
        
        for i in range(row_size):
             for j in range(degree):
               X_array[i][j]=X[i]**(1+j)
#        print("The array")
#        print(X_array)      
        return X_array


    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        #TODO
        
        Xpoly= self.polyfeatures(X, self.degree)
        
        # standardizing the dat a
        mean = np.mean(Xpoly, axis=0)
        std= np.std(Xpoly, axis=0)
        Xpoly= (Xpoly-mean) / std
        n,d= Xpoly.shape
        self.mean=mean 
        self.std= std 
        # regulairize by adding ones to the 0th column
        
        Xpoly = np.c_[np.ones((n,1)), Xpoly]
        
        ## initializing the theta since not in univariate data
#        initialize_theta= np.matrix(np.random.randn((d+1))).T
##        print("shaep of init_theta")
#        print(initialize_theta.shape)
        
#        iter = 15000
#        alpha = 0.000015
#        
#        fitted_model = LinearRegression(init_theta=initialize_theta , alpha=alpha, n_iter=iter)
#        fitted_model.fit(Xpoly,y,regLambda = self.regLambda)
##        print("shape of thetafitted")
##        print(fitted_model.theta.shape)
#        self.theta = fitted_model.theta
        
        regmatrix = self.regLambda* np.eye(d+1)
        regmatrix[0,0]= 0 
        
#        print(regmatrix.shape )
#        print(Xpoly.shape)
#        
        theta = np.linalg.pinv(Xpoly.T.dot(Xpoly) + regmatrix ).dot(Xpoly.T).dot(y)
        self.theta=theta
#        print("theta is")
        print(self.theta)
      
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        # TODO
        Xpoly= self.polyfeatures(X,self.degree)
        
        Xpoly= (Xpoly- self.mean)/ self.std
        n,d = Xpoly.shape
        Xpoly = np.c_[np.ones((n,1)), Xpoly]
#        print("Xploy shape is" )
#        print(Xpoly.shape)

        y_pred = Xpoly.dot(self.theta)
#        print("y_pred ")
#        print(y_pred.shape)

#        print(y_pred)
        return np.array(y_pred)



#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------


def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrains -- errorTrains[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTests -- errorTrains[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrains[0:1] and errorTests[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain);
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    for i in range(2, n):
        Xtrain_subset = Xtrain[:(i+1)]
        Ytrain_subset = Ytrain[:(i+1)]
        model = PolynomialRegression(degree, regLambda)
        model.fit(Xtrain_subset,Ytrain_subset)
        
        predictTrain = model.predict(Xtrain_subset)
        err = predictTrain - Ytrain_subset;
        errorTrain[i] = np.multiply(err, err).mean();
        
        predictTest = model.predict(Xtest)
        err = predictTest - Ytest;
        errorTest[i] = np.multiply(err, err).mean();
    
    return (errorTrain, errorTest)