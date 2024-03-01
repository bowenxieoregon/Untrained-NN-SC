import numpy as np 
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import os
from deepdecoders import deepdecoder
from earlystopping import EWMVar
from utils import get_tensor, cost_func,SRE,cost_func_norm
import random
from io import BytesIO
import wandb
import argparse


'''
set random seeds
'''
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

'''
set device
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
python run_non_deepdecoder.py --rho 0.1 --channel 256 --layers 6 --filters 3 --alpha 0.04 --patience 500 --lrnet 0.01 --lrvectors 0.0008 --maxite 70000 --order 2 --data_type strong_6_50_clear --saveto deepdecoder3.npy
'''



'''
set parameters
'''
parser = argparse.ArgumentParser(description='Some parameters.')
parser.add_argument('--rho', type=float, default=0.5, help='Sampling rate')
parser.add_argument('--channel', type=int, default=256, help='Number of channels')
parser.add_argument('--layers', type=int, default=5, help='Number of layers')
parser.add_argument('--filters', type=int, default=3, help='Filter size')
parser.add_argument('--alpha', type=float, default=0.1, help='Alpha')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--lrnet', type=float, default=0.01, help='Learning rate of net')
parser.add_argument('--lrvectors', type=float, default=0.01, help='Learning rate of C')
parser.add_argument('--maxite', type=int, default=3, help='max loop count')
parser.add_argument('--order', type=int, default=1, help='Order of run')
parser.add_argument('--data_type', type=str, default='strong_3_50_clear', help='Type of data')
parser.add_argument('--saveto', type=str, default='deepdecoder1.npy', help='Type of device')
args = parser.parse_args()
rho =  args.rho
channels = args.channel
layers = args.layers
filters = args.filters
alpha = args.alpha
patience = args.patience
lrnet = args.lrnet 
lrvectors = args.lrvectors
inner = args.maxite
data_type = args.data_type
savepath = args.saveto

'''
data loading
'''
post_str = '.npz'   
data_truth = np.load(data_type + post_str)
T_noise = data_truth['T']
T_raw = data_truth['T_true']
C_raw = data_truth['C_true']
S_raw = data_truth['S_true']
R = C_raw.shape[1]
print(f"we have {R} emitters")
T_true_noise = np.transpose(T_noise, (1,2,0))
T_true = np.transpose(T_raw, (1,2,0))
C_true = np.transpose(C_raw, (1,0))
S_true = np.transpose(S_raw, (2,1,0))




I,J,K = T_true.shape

'''
visualization band choice
'''
sel = 48

'''
NN architecture
'''
in_size = (16,16)
out_size = (I,I) #size of SLF
output_depth = 1
num_channels = [1] + [channels]*layers
net = deepdecoder(in_size,out_size,
                output_depth,
                num_channels=num_channels,
                need_sigmoid=True,
                filter_size=filters,
                last_noup=False,
               ).to(device) 
Z = torch.rand(R, in_size[0],in_size[1]).unsqueeze(1).to(device)
Z.requires_grad = False

'''
sampling
'''
IJ = I*J
num_samples = round(rho*IJ)
Omega = random.sample(range(IJ), num_samples)
Ov = np.zeros(IJ, dtype=bool)
Ov[Omega] = True
Om = Ov.reshape(I,J,order = 'F')
W = torch.from_numpy(Om).type(torch.float32)
Wx = W
Wxx = Wx.unsqueeze(dim = -1)
Wxxx = Wxx.repeat(1,1,K).bool()
Wxxx = Wxxx.to(device).requires_grad_(False)


T_true_noise_pytorch = torch.from_numpy(np.log10(T_true_noise)).type(torch.float32).to(device).requires_grad_(False)
T_max = T_true_noise_pytorch[Wxxx].max().requires_grad_(False)
T_min = T_true_noise_pytorch[Wxxx].min().requires_grad_(False)
T_normalized = (T_true_noise_pytorch - T_min) / (T_max - T_min) #normalized data
T_normalized.requires_grad = False



'''
initialize early stopping
'''
alpha = alpha
patience = patience
es = EWMVar(alpha, patience)

'''
initialize C and S
'''
shapec = (R,K)
C_est = np.random.rand(*shapec)
C_pytorch = torch.from_numpy(C_est).to(device).requires_grad_(True)
S_est = net(Z).squeeze()

com_proj = 'DeepDecoder-'
run = wandb.init(
    # Set the project where this run will be logged
    project= com_proj + data_type,
    name = "run" + str(args.order),
)

X_true = torch.from_numpy(T_true).to(device).requires_grad_(False) #ground truth

'''
optimizer setting
'''
optimizer = torch.optim.Adam([
    {'params':net.parameters(), 'lr': lrnet},
    {'params':C_pytorch, 'lr': lrvectors}
])

def adjust_lr(epoch, base_lr, epoch_target=10000, decay_interval=3500, decay_rate=0.95, target_lr=0.0005):
    if epoch < epoch_target:
        return decay_rate ** (epoch // decay_interval)
    else:
        epochs_after_target = (epoch - epoch_target) // decay_interval
        decay_factor_after_target = decay_rate ** epochs_after_target
        return (target_lr / base_lr) * decay_factor_after_target

lambda1 = lambda epoch: adjust_lr(epoch, lrnet, 20000, 2000, 0.65, 0.00001)
lambda2 = lambda epoch: adjust_lr(epoch, lrvectors, 20000, 2000, 0.65, 0.000008)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])


# lambda1 = lambda epoch: 0.9 ** (epoch // 2000)
# lambda2 = lambda epoch: 0.9 ** (epoch // 2000)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
#C_pytorch: 0.0008 lr is good

for i in range(inner):
    optimizer.zero_grad()
    slf_complete = net(Z).squeeze()
    X_from_slf = get_tensor(slf_complete, torch.log(1 + torch.exp(C_pytorch))) 
    loss = cost_func_norm(T_normalized, X_from_slf, Wxxx)
    print(f"ite {i}")
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        slf_complete_temp = net(Z).detach().clone().squeeze()
        X_from_slf_temp = get_tensor(slf_complete_temp.to(device), torch.log(1 + torch.exp(C_pytorch.detach().clone())))
        es.update_av(X_from_slf_temp.cpu().numpy() * (T_max.item() - T_min.item()) + T_min.item(),i)
        mse, sre = SRE(torch.from_numpy(T_true).to(device),torch.pow(10,X_from_slf_temp * (T_max - T_min) + T_min))
        var = es.emv
        log_coss = cost_func_norm(T_normalized, X_from_slf_temp, Wxxx).item()
        wandb.log({"Log-Loss": log_coss, "SRE": sre, "var": var})
        if i % 10 == 0:
            data = np.power(10, X_from_slf_temp.cpu().numpy() * (T_max.item() - T_min.item()) + T_min.item())
            np.save(savepath, data)
    
    scheduler.step()