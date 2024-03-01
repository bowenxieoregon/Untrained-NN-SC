import torch
import torch.nn as nn
from deepdecoders import deepdecoder
import numpy as np
from utils import matrix_normalize, outer, get_tensor, cost_func,NAE, SRE
from torch.optim.lr_scheduler import StepLR
from earlystopping import EWMVar
import os
import wandb 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
use deepdecoder to optimize S
'''
def optimize_s_dip1(X_true, W, X, Z1, C, R, model_path,ES,ite_cu,alpha, patience, channels, layers, filters, lr, loop_count = 3):
    """
    Arguments:
        W : Mask 
        X : sampled tensor
        Z1 : current latent vectors estimate for R emitters
        C : current psd estimate

    Returns:
        the updated S estimate
    """
    alpha = alpha
    patience = patience
    ESES = EWMVar(alpha=alpha,p=patience)
    X_true  = torch.from_numpy(X_true).to(device)
    ite = ite_cu
    ESES = ES
    channels = channels
    layers = layers
    filters = filters
    lr = lr

    W = torch.from_numpy(W).type(torch.float32)
    X = torch.from_numpy(X).type(torch.float32).to(device)
    Z_est = Z1
    Z_est.requires_grad_(True)
    C = torch.from_numpy(C).type(torch.float32).to(device)
    R = int(R)
    K = X.shape[2]
    Wx = W
    Wxx = Wx.unsqueeze(dim = -1)
    Wxxx = Wxx.repeat(1,1,K)
    Wxxx[Wxxx<0.5] = 0
    Wxxx[Wxxx>=0.5] = 1
    Wxxx = Wxxx.to(device)

    #in_size = (1,256) #size of latent code
    in_size = (16,16)
    out_size = (51,51)
    #out_size = (51,51) #size of SLF
    output_depth = 1
    num_channels = [1] + [channels]*layers

    decoded = deepdecoder(in_size,out_size,
                    output_depth,
                    num_channels=num_channels,
                    need_sigmoid=True,
                    #need_tanh=True,
                    filter_size=filters,
                    last_noup=False,
                ).to(device)
    
    decoded.load_state_dict(torch.load(model_path))
    slf_complete = decoded(Z_est).squeeze()
    
    '''
    decoded = Decoderonly_sa()
    decoded.load_state_dict(torch.load(model_path))
    slf_complete = decoded(Z_est).squeeze()
   # X_from_slf = get_tensor(slf_complete[:,0,:,:], C)
    '''
    X_from_slf = get_tensor(slf_complete, C)
    
    # with torch.no_grad():
    #     mse, sre = SRE(X_true,X_from_slf)
    #     var = ESES.emv
    #     wandb.log({"SRE": sre, "var": var, "mse":mse})
    
    optimizer = torch.optim.Adam(decoded.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.99)

    for i in range(loop_count):
        optimizer.zero_grad()
        slf_complete = decoded(Z_est).squeeze()
        X_from_slf = get_tensor(slf_complete, C)
        #print(f"X shape is {X.shape}, X_from_slf shape is {X_from_slf.shape}")
        loss = cost_func(X, X_from_slf, Wxxx)#+ lambda_reg*torch.norm(Z_est)
        with torch.no_grad():
            wandb.log({"loss": loss.item()})
        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     wandb.log({"loss": loss.item()})
        print(ite_cu + i)
        ite = ite + 1
        scheduler.step()

    torch.save(decoded.state_dict(), model_path)
    return slf_complete, Z_est, ESES


def optimize_s_dip1_div(X_true, W, X, Z1, C, R, model_path,ES,ite_cu,alpha, patience, channels, layers, filters, lr, loop_count = 3):
    """
    Arguments:
        W : Mask 
        X : sampled tensor
        Z1 : current latent vectors estimate for R emitters
        C : current psd estimate

    Returns:
        the updated S estimate
    """
    alpha = alpha
    patience = patience
    ESES = EWMVar(alpha=alpha,p=patience)
    X_true  = torch.from_numpy(X_true).to(device)
    ite = ite_cu
    ESES = ES
    channels = channels
    layers = layers
    filters = filters
    lr = lr

    W = torch.from_numpy(W).type(torch.float32)
    X = torch.from_numpy(X).type(torch.float32).to(device)
    Z_est = Z1
    Z_est.requires_grad_(True)
    C = torch.from_numpy(C).type(torch.float32).to(device)
    R = int(R)
    K = X.shape[2]
    Wx = W
    Wxx = Wx.unsqueeze(dim = -1)
    Wxxx = Wxx.repeat(1,1,K)
    Wxxx[Wxxx<0.5] = 0
    Wxxx[Wxxx>=0.5] = 1
    Wxxx = Wxxx.to(device)

    #in_size = (1,256) #size of latent code
    in_size = (16,16)
    out_size = (51,51) #size of SLF
    output_depth = 1
    num_channels = [1] + [channels]*layers

    NET = []
    for r in range(R):
        decoded = deepdecoder(in_size,out_size,
                        output_depth,
                        num_channels=num_channels,
                        need_sigmoid=True,
                        #need_tanh=True,
                        filter_size=filters,
                        last_noup=False,
                    ).to(device)
        decoded.load_state_dict(torch.load(model_path + str(r) + '.pth'))
        NET.append(decoded)
    slf_complete = []
    for r in range(R):
        slf_complete.append(NET[r](Z_est[r].unsqueeze(0)))
    slf_complete = torch.stack(slf_complete,dim = 0).squeeze()
    
    '''
    decoded = Decoderonly_sa()
    decoded.load_state_dict(torch.load(model_path))
    slf_complete = decoded(Z_est).squeeze()
   # X_from_slf = get_tensor(slf_complete[:,0,:,:], C)
    '''
    X_from_slf = get_tensor(slf_complete, C)
    
    # with torch.no_grad():
    #     mse, sre = SRE(X_true,X_from_slf)
    #     var = ESES.emv
    #     wandb.log({"SRE": sre, "var": var, "mse":mse})
    
    optimizer = torch.optim.Adam(decoded.parameters(), lr=lr)
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for i in range(loop_count):
        optimizer.zero_grad()
        slf_complete = []
        for r in range(R):
            slf_complete.append(NET[r](Z_est[r].unsqueeze(0)))
        slf_complete = torch.stack(slf_complete,dim = 0).squeeze()
        X_from_slf = get_tensor(slf_complete, C)
        loss = cost_func(X, X_from_slf, Wxxx)#+ lambda_reg*torch.norm(Z_est)
        loss.backward()
        optimizer.step()
        print(ite_cu + i)
        ite = ite + 1
        #scheduler.step()

    for r in range(R):
        torch.save(NET[r].state_dict(), model_path + str(r) + '.pth')
    return slf_complete, Z_est, ESES