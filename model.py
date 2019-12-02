# python3 代码
import import_path
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

class DeepMultiTagPredictor(torch.nn.Module):
    """ @ mutable | tunable
        
        the module to use the high_feature to predict the tag.
        and get the loss for error-proporganda
        the module simply model the task as multi binary logistic regression classifiers
    """
    def __init__(self, aesthetic_feature_dim, label_num=14):
        self.aesthetic_feature_dim = aesthetic_feature_dim
        self.label_num = 14
        self.weight_w = torch.nn.Parameter(torch.Tensor(self.label_num, self.aesthetic_feature_dim))
        self.weight_w = torch.nn.init.normal_(self.wei_user, mean=0.0, std=1.0)

    def forward(self, high_feature):
        """
            use the high_feature to predict the label class
            
            @ high_feature: the high level feature wants to predict the tag
            @ return      : numpy([batch_size, labels_num]) #prob [0, 1]
        """
        output = torch.matmul(high_feature, self.weight_w.transpose())
        return torch.nn.Sigmoid(output) # simple

class AestheticFeatureLayer(torch.nn.Module):
    """ @ mutable | tunable

        Extract the high level aesthetic feature from tags prediction 
    """
    def __init__(self, cnn_feat_dim, aesthetic_feature_dim, batch_size):
        self.bs = batch_size
        self.dim= cnn_feat_dim
        self.fdim=aesthetic_feature_dim
        # ====================== inner parameters ====================
        self.layers = [cnn_feat_dim, 500, aesthetic_feature_dim]
        self.activiate = torch.nn.Tanh()
        # ====================== inner initialize ====================
        self.mlp = []
        for prev, nex in zip(self.layers, self.layers[1:]):
            self.mlp.append(torch.nn.Linear(prev, nex))
            self.mlp.append(self.activiate)

    def forward(self, cnn_feature):
        """
            @cnn_feature: np.array()  .shape = (self.bs, self.dim)
            @return:      np.array()  .shape = (self.bs, self.fdim)
        """ 
        high_feature = self.transform_function(cnn_feature)
        return high_feature
    
    def transform_function(self, raw_feature): 
        """ @ overridable

            this function is the transform process of the raw_feature

            @raw_feature: np.array()  .shape = (self.bs, self.dim)
            @return     : np.array()  .shape = (self.bs, self.fdim)
        """
        # TODO (try different function, the basic is to use the 2-layer MLP )
        tensor_flow = raw_feature
        for layer in self.mlp:
            output = layer(tensor_flow)
        return tensor_flow
        

class Cond_LSTM(torch.nn.Module):
    """
        [U diag(Fs) V] . shape = (n_hidden, n_hidden) => n_F 为 
        
    """
    def __init__ (self, n_input, n_hidden, n_F, n_cond_dim):
        super(Cond_LSTM, self).__init__()
        self.wei_U = torch.nn.ParameterList()
        self.wei_V = torch.nn.ParameterList()
        self.wei_WI = torch.nn.ParameterList()
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.n_cond_dim = n_cond_dim
        self.n_F = n_F
        self.wei_F = torch.nn.Parameter(torch.Tensor(n_F, n_cond_dim))
        for i in range(4): # responding to "i f g o" gates respectively
            self.wei_U.append(torch.nn.Parameter(torch.Tensor(n_hidden, n_F)))
            self.wei_V.append (torch.nn.Parameter(torch.Tensor(n_F, n_hidden)))
            self.wei_WI.append ( torch.nn.Parameter(torch.Tensor(n_hidden, n_input)))      # for input weight
        
        self.init_parameters()

    def init_parameters(self): # FIXME(是否有更加好的实现方式)
        self.wei_F     = init.normal_(self.wei_F, mean=0.0, std=1.0)
        for i in range(4):
            self.wei_U[i]  = init.normal_(self.wei_U[i], mean=0.0, std=1.0)
            self.wei_V[i]  = init.normal_(self.wei_V[i], mean=0.0, std=1.0)
            self.wei_WI[i] = init.normal_(self.wei_WI[i], mean=0.0, std=1.0)
    
    def forward(self, tnsr_input, tpl_h0_c0, tnsr_cond): # XXX 每个batch必须要cond相同
        """
        @ tnsr_input : (n_step, n_batch, n_feat_dim)
        @ tpl_h0_c0  : ((n_batch, n_hidden) , (n_batch, n_hidden)) 一个tuple
        @ tnsr_cond  : (n_cond_dim, ), the same for the whole batch
        """
        assert(tnsr_cond.shape[0] == self.n_cond_dim)
        tnsr_cond = tnsr_cond.unsqueeze(1)
        wei_WH = [ self.wei_U[i].matmul((self.wei_F.matmul(tnsr_cond))*self.wei_V[i])  for i in range(4) ] # shape = (n_hidden, n_hidden)

        n_batch = tnsr_input.shape[1]
        n_step  = tnsr_input.shape[0]
        c = [ tpl_h0_c0[1] ]  # self.c[i].shape = (n_hidden, n_batch)
        assert (c[0].shape == (self.n_hidden, n_batch))
        h = [ tpl_h0_c0[0] ]  # self.c[i].shape = (n_hidden, n_batch)
        assert (h[0].shape == (self.n_hidden, n_batch))
        
        for t in range(n_step): # TODO add Bias, bi and bh
            it = torch.sigmoid(self.wei_WI[0].matmul(tnsr_input[t].t()) + wei_WH[0].matmul(h[t]))  # it.shape = (n_hidden, n_batch)
            ft = torch.sigmoid(self.wei_WI[1].matmul(tnsr_input[t].t()) + wei_WH[1].matmul(h[t]))  
            gt = torch.tanh(self.wei_WI[2].matmul(tnsr_input[t].t()) + wei_WH[2].matmul(h[t]))
            ot = torch.sigmoid(self.wei_WI[3].matmul(tnsr_input[t].t()) + wei_WH[3].matmul(h[t]))
            
            c.append (ft * c[t] + it * gt)
            h.append (ot * torch.tanh(c[t+1]))
            
        assert (len(h) == len(c) and len(c) == n_step+1)
        assert (h[0].shape == (self.n_hidden, n_batch)) # 列向量
        return h, c

    # XXX 不要有多个batch，不要梯度
    def eval_start(self, tpl_h0_c0, tnsr_cond):
        """ eval_start 然后每个 eval_step 输出一个output
        @tpl_h0_c0 : (h0, c0)   type(h0|c0) =  torch.tensor ;  h0|c0.shape = (n_hidden,)
        @tnsr_cond : tensor shape=(n_cond)
        """
        assert(tnsr_cond.shape[0] == self.n_cond_dim)
        tnsr_cond = tnsr_cond.unsqueeze(1)
        self.wei_WH = [ self.wei_U[i].matmul((self.wei_F.matmul(tnsr_cond))*self.wei_V[i])  for i in range(4) ]
        self._eval_c = tpl_h0_c0[1].unsqueeze(1)  # self.c[i].shape = (n_hidden, 1)
        assert (self._eval_c.shape == (self.n_hidden, 1))
        self._eval_h = tpl_h0_c0[0].unsqueeze(1)  # self.c[i].shape = (n_hidden, n_batch)
        assert (self._eval_h.shape == (self.n_hidden, 1))
        pass

    def eval_step (self, tnsr_input):
        """ eval_start 然后每个 eval_step 输出一个output

        @ tnsr_input : tensor shape=(input_feat_dims)
        """
        tnsr_input = tnsr_input.unsqueeze(1)  # make it bacame matrix
        it = torch.sigmoid(self.wei_WI[0].matmul(tnsr_input) + self.wei_WH[0].matmul(self._eval_h))  # it.shape = (n_hidden, 1)
        ft = torch.sigmoid(self.wei_WI[1].matmul(tnsr_input) + self.wei_WH[1].matmul(self._eval_h))  
        gt = torch.tanh(self.wei_WI[2].matmul(tnsr_input) + self.wei_WH[2].matmul(self._eval_h))
        ot = torch.sigmoid(self.wei_WI[3].matmul(tnsr_input) + self.wei_WH[3].matmul(self._eval_h))
            
        self._eval_c = ft * self._eval_c + it * gt
        self._eval_h = ot * torch.tanh(self._eval_c)

        return self._eval_h.squeeze(), self._eval_c.squeeze()
