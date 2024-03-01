import torch
import numpy as np
import tensorly as tl
from scipy.optimize import nnls
from munkres import Munkres
from numpy.linalg import norm
import os
import random


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)



# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ColumnNormalization(X):
    #UNTITLED normalize each column of X
    #  To make its 2-norm to 1
    X = np.array(X)
    m,n=X.shape
    Y = np.zeros((m,n))
    D = np.zeros(n)
    for i in range(n):
        D[i] = np.linalg.norm(X[:,i])
        if D[i] == 0:
            Y[:,i] = X[:,i]
        else:
            Y[:,i] = X[:,i]/D[i]
    return Y

def matrix_normalize(matrix):
    norms = torch.norm(matrix,p = 'fro')
    return matrix/norms

def outer(mat, vec):
    prod = torch.zeros(( *mat.shape, *vec.shape), dtype=torch.float32).to(device)
    for i in range(len(vec)):
        prod[:,:,i] = mat * vec[i]
    return prod

def get_tensor(S, C):
    prod = 0
    for i in range(C.shape[0]):
        prod += outer(S[i,:,:], C[i])
    return prod

# def get_tensor_tl(S,C):
#     # Get the tensor outer product 
#     sizec = C.shape
#     X = np.zeros((51,51,sizec[1]))
#     for r in range(sizec[0]):
#         X = X + tl.tenalg.outer((S[r],C[r])) 
#     return X 

def cost_func(X, X_from_slf, Wx):
    return torch.norm((Wx * torch.log10(X)) - (Wx * torch.log10(X_from_slf)), p = 'fro')**2


def cost_func_nolog(X, X_from_slf, Wx):
    return torch.norm(((Wx*X) - (Wx* X_from_slf)), p = 'fro')**2

def cost_func_norm(T, X_from_slf, Wx):
    return torch.norm((Wx * T) - (Wx * X_from_slf), p = 'fro')**2



def nnls_my(H, Y):
    """Non-negative least squares by the active set method,
    solves min ||Y - CH||_F, s.t. C >= 0, all equivalently,
    solves min ||Y.T - H.T @ C.T ||_F, s.t. C >= 0.

    return C
    """
    xi,xj = H.T.shape
    yi, yj = Y.T.shape
    C = np.zeros((xj,yj)).T
    
    for k in range(yj):
        this_Y = Y.T[:, k]
        # densify the column vector if sparse
        if hasattr(this_Y, 'toarray'):
            this_Y = this_Y.toarray().squeeze()
        C.T[:, k], this_res = nnls(H.T, this_Y)
    return C


def col_normalization_l1(W):
    norms = np.linalg.norm(W, axis=0,ord = 1)
    norms[norms==0] = 1
    W_normalized = W / norms[None,:]
    return W_normalized

'''
This is actually computing NAE between W and West (1-norm)
'''
def NAE(W,West):
    r = W.shape[1]
    W_temp = col_normalization_l1(W)
    West_temp = col_normalization_l1(West)
    W_expanded = W_temp[:,:,None]
    West_expanded = West_temp[:,None,:]
    DIST = np.sum(np.abs(W_expanded - West_expanded), axis=0)
    m = Munkres()
    indexes = m.compute(DIST)
    print(indexes)
    row_indexes, col_indexes = zip(*indexes)
    W_reordered = W[:,row_indexes]
    West_reordered = West[:,col_indexes]
    return W_reordered, West_reordered

def SRE(W, West): #sre calculation
    mse = torch.norm(W - West, p='fro')**2
    norm_w = torch.norm(W, p='fro')**2
    return mse, mse/norm_w



