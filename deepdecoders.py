import numpy as np 
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torchsummary import summary
import random

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


'''
The following code is from model proposed in 'https://github.com/MLI-lab/cs_deep_decoder'
'''

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

def conv(in_f, out_f, kernel_size, stride=1, pad='zero',bias=False):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)        

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)

def deepdecoder(
        in_size,
        out_size,
        num_output_channels=1, 
        num_channels=[256]*5, 
        filter_size=3,
        need_sigmoid=True,
        pad ='reflection', 
        upsample_mode = 'bilinear', 
        act_fun= nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        bias=False,
        last_noup=False, # if true have a last extra conv-relu-bn layer without the upsampling before linearly combining them
        ):
    
    depth = len(num_channels)
    scale_x,scale_y = (out_size[0]/in_size[0])**(1./depth), (out_size[1]/in_size[1])**(1./depth)
    hidden_size = [(int(np.ceil(scale_x**n * in_size[0])),
                    int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, depth)] + [out_size]
    
    #print(hidden_size)
    
    num_channels = num_channels + [num_channels[-1],num_channels[-1]]
    
    n_scales = len(num_channels) 
    
    if not (isinstance(filter_size, list) or isinstance(filter_size, tuple)) :
        filter_size   = [filter_size]*n_scales
    
    model = nn.Sequential()

    for i in range(len(num_channels)-2):
        model.add(conv( num_channels[i], num_channels[i+1],  filter_size[i], 1, pad = pad, bias=bias))
        if upsample_mode!='none' and i != len(num_channels)-2:
            # align_corners: from pytorch.org: if True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. Default: False
            # default seems to work slightly better
            model.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode,align_corners=False))
        
        if(bn_before_act): 
            model.add(nn.BatchNorm2d( num_channels[i+1] ,affine=bn_affine))
        if act_fun is not None:    
            model.add(act_fun)
        if not bn_before_act:
            model.add(nn.BatchNorm2d( num_channels[i+1], affine=bn_affine))
    
    if last_noup:
        model.add(conv( num_channels[-2], num_channels[-1],  filter_size[-2], 1, pad = pad, bias=bias))
        model.add(act_fun)
        model.add(nn.BatchNorm2d( num_channels[-1], affine=bn_affine))
    
    model.add(conv( num_channels[-1], num_output_channels, 1, pad = pad,bias=bias))
    if need_sigmoid:
        model.add(nn.Sigmoid())
   
    return model

'''
inp shape is torch.Size([4, 1, 256])
OUT shape is torch.Size([4, 51, 51])
'''
