import torch
import torch.nn as nn
from models import *
import numpy as np
from utils import matrix_normalize, outer, get_tensor, cost_func,NAE, SRE, cost_func_origin
from torch.optim.lr_scheduler import StepLR
from earlystopping import EWMVar
import os
import wandb

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize_s_dip3(X_true, W, X, Z1, C, R, model_path,ite_cu,alpha, patience, channels, layers, filters, lr, loop_count, param_vec):
    """
    Arguments:
        W : Mask 
        X : sampled tensor
        Z1 : current latent vectors estimate for R emitters
        C : current psd estimate

    Returns:
        the updated S estimate
    """

    param_vector = param_vec
    # alpha = alpha
    # patience = patience
    # ESES = EWMVar(alpha=alpha,p=patience)
    X_true  = torch.from_numpy(X_true).to(device)
    ite = ite_cu
    channels = channels
    layers = layers
    filters = filters
    lr = lr

    W = torch.from_numpy(W).type(torch.float32)
    X = torch.from_numpy(X).type(torch.float32).to(device)
    Z_est = Z1
    Z_est.requires_grad_(False)
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
    # in_size = (51,51)
    input_depth = 1
    NET_TYPE = 'skip'
    pad = 'zero'
    upsample_mode = 'nearest'
    decoded = get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride').to(device)

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
    # optimizer = torch.optim.SGD(decoded.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.99)

    for i in range(loop_count):
        optimizer.zero_grad()
        slf_complete = decoded(Z_est).squeeze()
        X_from_slf = get_tensor(slf_complete, C)
        loss = cost_func_origin(X, X_from_slf, Wxxx) #+ lambda_reg*torch.norm(Z_est)\



        '''
        new add
        '''
        with torch.no_grad():
            state_dict = decoded.parameters() 
            param_vector_new = torch.cat([param.flatten() for param in state_dict])
            wandb.log({"distance": torch.norm(param_vector_new - param_vector)})


        # with torch.no_grad():
        #     # wandb.log({"loss": loss})
        loss.backward()
        optimizer.step()
        print(ite_cu + i)
        ite = ite + 1
        scheduler.step()

    torch.save(decoded.state_dict(), model_path)
    return slf_complete, Z_est