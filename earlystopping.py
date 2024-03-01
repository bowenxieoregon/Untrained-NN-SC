import torch 
import torch.nn as nn 
import numpy as np

class EWMVar():
    def __init__(self, alpha, p):
        self.alpha = alpha
        self.patience = p
        self.wait_count = 0
        self.best_emv = float('inf')
        self.best_epoch = 0
        self.stop = False
        self.ema = None
        self.emv = None

    def check_stop(self, cur_epoch):
      #stop when EMV doesn't decrease for consecutive P(patience) times
        if self.emv < self.best_emv:
            self.best_emv = self.emv
            self.best_epoch = cur_epoch
            self.wait_count = 0
        else:
            self.wait_count += 1
            self.stop = self.wait_count >= self.patience

    def update_av(self, cur_img, cur_epoch):
        #initialization
        if cur_epoch == 0:
            self.ema = cur_img
            self.emv = 0
        #update
        else:
            delta = cur_img - self.ema
            tmp_ema = self.ema + self.alpha * delta
            self.ema = tmp_ema
            self.emv = (1 - self.alpha) * (self.emv + self.alpha * (np.linalg.norm(delta) ** 2))