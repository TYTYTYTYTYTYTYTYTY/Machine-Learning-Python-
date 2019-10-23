import numpy as np

"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

def l2distance(X,Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
    
    
    
    # dist = np.linalg.norm(a-b)
    
    G = X.T.dot(Z)

    S = np.sum(X**2, axis =0)
    S = np.tile(S.reshape(-1,1),m)


    R = np.sum(Z**2, axis =0)
    R= np.tile(R.reshape(1,-1),(n,1))

    D2 = S-2*G+R
    D2[D2<0] =0
    
    D = np.sqrt(D2)


    
    
    return D
