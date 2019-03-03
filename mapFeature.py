import numpy as np
def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
 
    
    deg = 6 
    X=np.ones((x2.shape[0],27))
#    X=[]
    col = 0
    
    for i in range(1,deg+1):
        for j in range(0,i+1):
           X[:,col]= x1**(i-j) * x2**(j)
           col = col +1 
           
#    X = X.T
#    X = X[ :, 1:]
#    
    print(X.shape)
    return X
#   testing      
#    import numpy as np
#
#    degree = 6
#    out = np.ones(( x1.shape[0], sum(range(degree + 2)) )) # could also use ((degree+1) * (degree+2)) / 2 instead of sum
#    curr_column = 1
#    for i in range(1, degree + 1):
#        for j in range(i+1):
#            out[:,curr_column] = np.power(x1,i-j) * np.power(x2,j)
#            curr_column += 1
#        
#    print(out.shape)
#    return out
##
#X= np.asarray([ 1,2 ,3 ,4]).T
#Y = np.asarray([ 1 ,2, 3,4]).T
###X = np.asarray(X)
###
###Y = np.asarray(Y)
#p = mapFeature(X,Y)


#
#print(p)