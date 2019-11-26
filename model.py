# python3 代码
import numpy as np
import torch
import os
import pandas as pd
import xkcv_optimizer
# using ClassifierChain
# from skmultilearn.problem_transform import 
from torch import *
import torch.nn.init as init
from nlp_score import score # 评分函数

class Cond_LSTM(torch.nn.Module):
    """
        [U diag(Fs) V] . shape = (n_hidden, n_hidden) => n_F 为 
        
    """
    def __init__ (self, n_input, n_hidden, n_F, n_cond_dim):
        super(Cond_LSTM, self).__init__()
        self.wei_U = []
        self.wei_V = []
        self.wei_W = []
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.n_cond_dim = n_cond_dim
        self.n_F = n_F
        for i in range(4): # responding to "i f g o" gates respectively
            self.wei_U[i]  = torch.Parameter(torch.Tensor(n_hidden, n_F))
            self.wei_V[i]  = torch.Parameter(torch.Tensor(n_F, n_hidden))
            self.wei_F     = torch.Parameter(torch.Tensor(n_F, n_cond_dim))
            self.wei_WI[i] = torch.Parameter(torch.Tensor(n_hidden, n_input))      # for input weight
        
        self.init_parameters()

    def init_parameters(self): # FIXME(是否有更加好的实现方式)
        for i in range(4):
            self.wei_U[i]  = init.normal_(self.wei_U[i], mean=0.0, std=1.0)
            self.wei_V[i]  = init.normal_(self.wei_V[i], mean=0.0, std=1.0)
            self.wei_F     = init.normal_(self.wei_F[i], mean=0.0, std=1.0)
            self.wei_WI[i] = init.normal_(self.wei_WI[i], mean=0.0, std=1.0)
    
    def forward(self, tnsr_input, tpl_h0_c0, tnsr_cond): # XXX 每个batch必须要cond相同
        """
        @ tnsr_input : (n_step, n_batch, n_feat_dim)
        @ tpl_h0_c0  : ((n_batch, n_hidden) , (n_batch, n_hidden)) 一个tuple
        @ tnsr_cond  : (n_cond_dim, ), the same for the whole batch
        """
        assert(tnsr_cond.shape[0] == self.n_cond_dim)
        wei_WH = [ self.wei_U.matmul((self.F.matmul(tnsr_cond))*self.wei_V[i])  for i in range(4) ]

        n_batch = tnsr_input.shape[2]
        n_step  = tnsr_input.shape[0]
        c = [ tpl_h0_c0[1]]  # self.c[i].shape = (n_hidden, n_batch)
        assert (c[0].shape == (self.n_hidden, n_batch))
        h = [ tpl_h0_c0[0]]  # self.c[i].shape = (n_hidden, n_batch)
        assert (h[0].shape == (self.n_hidden, n_batch))
        
        for t in range(n_step): # TODO add Bias, bi and bh
            it = torch.sigmoid(self.wei_WI[0].matmul(tnsr_input[t]) + wei_WH[0].matmul(h[t]))  # it.shape = (n_hidden, n_batch)
            ft = torch.sigmoid(self.wei_WI[1].matmul(tnsr_input[t]) + wei_WH[1].matmul(h[t]))  
            gt = torch.tanh(self.wei_WI[2].matmul(tnsr_input[t]) + wei_WH[2].matmul(h[t]))
            ot = torch.sigmoid(self.wei_WI[3].matmul(tnsr_input[t]) + wei_WH[3].matmul(h[t]))
            
            c[t+1] = ft * c[t] + it * gt
            h[t+1] = ot * torch.tanh(c[t+1])
            
        assert (len(h) == len(c) && len(c) == step+1)
        assert (h[0].shape == (n_hidden, n_batch)) # 列向量
        return h, c

    # XXX 不要有多个batch，不要梯度
    def eval_start(self, tpl_h0_c0, tnsr_cond):
        """ eval_start 然后每个 eval_step 输出一个output
        @tpl_h0_c0 : (h0, c0)   type(h0|c0) =  torch.tensor ;  h0|c0.shape = (n_hidden,)
        @tnsr_cond : tensor shape=(n_cond)
        """
        assert(tnsr_cond.shape[0] == self.n_cond_dim)
        wei_WH = [ self.wei_U.matmul((self.F.matmul(tnsr_cond))*self.wei_V[i])  for i in range(4) ]
        self._eval_c = [ tpl_h0_c0[1]]  # self.c[i].shape = (n_hidden, n_batch)
        assert (self._eval_c[0].shape == (self.n_hidden, ))
        self._eval_h = [ tpl_h0_c0[0]]  # self.c[i].shape = (n_hidden, n_batch)
        assert (self._eval_h[0].shape == (self.n_hidden, ))
        pass

    def eval_step (self, tnsr_input)
        """ eval_start 然后每个 eval_step 输出一个output

        @ tnsr_input : tensor shape=(input_feat_dims)
        """
        tnsr_input = tnsr_input.unsqueeze(1)  # make it bacame matrix
        it = torch.sigmoid(self.wei_WI[0].matmul(tnsr_input) + wei_WH[0].matmul(self._eval_h[t]))  # it.shape = (n_hidden, 1)
        ft = torch.sigmoid(self.wei_WI[1].matmul(tnsr_input) + wei_WH[1].matmul(self._eval_h[t]))  
        gt = torch.tanh(self.wei_WI[2].matmul(tnsr_input) + wei_WH[2].matmul(self._eval_h[t]))
        ot = torch.sigmoid(self.wei_WI[3].matmul(tnsr_input) + wei_WH[3].matmul(self._eval_h[t]))
            
        self._eval_c = ft * self._eval_c + it * gt
        self._eval_h = ot * torch.tanh(self._eval_c)

        return self._eval_h.squeeze(), self._eval_c.squeeze()
