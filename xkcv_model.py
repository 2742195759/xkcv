## python3 代码
import numpy as np
import torch
import pandas as pd
import xkcv_optimizer
from torch import *
import torch.nn.init as init
from nlp_score import score # 评分函数


def get_instance(name, args):
    return eval(name)(args)

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
            
# XXX (Driver Module)
# XXX (需要注意，句子要加入 <SOS> 和 <EOS> 在句首和句尾
class User_Caption(nn.Module):
    def __init__(self, args) :
        super(User_Caption, self).__init__()
        self.batch_size = args.batch_size
        self.device = args.device
        self.n_user = args.n_user
        self.n_user_dim = args.n_user_dim
        self.n_img_dim = args.n_img_dim   # [int] img feature dim
        self.n_word_dim = args.n_word_dim
        self.n_voc_size = args.n_voc_size
        self.wei_user = nn.Parameter(Tensor(n_user, n_user_dim))
        self.reset_parameter()
        self.word_emb = nn.Embedding(self.n_voc_size, n_word_dim)
        # construct the link to the lstm
        self.cond_lstm = Cond_LSTM(n_word_dim, args.n_hidden, args.n_F, args.n_user_dim)
        self.n_hidden = args.n_hidden
        self.wei_WI = nn.Parameter(Tensor(self.n_img_dim, args.hidden))   # self.wei_WI 见图, image_feat transform matrix
        self.wei_WP = nn.Parameter(Tensor(args.hidden, args.n_voc_size))  # self.wei_WP 预测softmax的地方，

        self.optimizer = xkcv_optimizer.get_instance(self, args) # XXX 一定要在 所有parameter之后
        # XXX 初始化变量要reset_parameter中添加

    def reset_parameter(self):
        self.wei_user = init.normal_(self.wei_user, mean=0.0, std=1.0)
        self.wei_WI   = init.normal_(self.wei_WI, mean=0.0, std=1.0)
        self.wei_WP   = init.normal_(self.wei_WP, mean=0.0, std=1.0)
        # self.

    def _step_loss(self, input_batch): # TODO 将这个拆分为 forward 和 loss_fn 两个
        """
        输入格式: 
        @ input_batch : 
            type(input_batch) = space @后为关键字
            @ key [int]   uid      : 0-base
            @ key [numpy] img_feat : .shape = (n_batch, n_img_dim)
            @ key [numpy] cap_seq  : .shape = (n_batch, n_max_len)
        """
        self.train()

        user_embedding = self.wei_user[input_batch.uid]
        img_feat = input_batch.img_feat
        n_batch = img_feat.shape[0]
        lstm_input = torch.transpose(self.word_emb(input_batch.cap_seq), 0, 1) # n_step, n_batch, dim
        c0 = torch.randn(self.n_hidden).repeat(n_batch,1).transpose()  # TODO (???变为Parameter吗需要?)
        h0 = torch.from_numpy(img_feat).matmul(self.wei_WI)
        h, c = self.cond_lstm(lstm_input, (h0,c0)) 
        softmax = torch.nn.Softmax(dim=-1)
        loss = torch.tensor(0) # 初始化为0, 否则
        for i, ht in enumerate(h[1:]): # TODO(check) <EOS>怎么考虑,然后Loss计算要考虑后面的吗。
            indexes = (range(n_batch), input_batch.cap_seq[:,i])  # 同一个step中，所有的batch的gt对应的prob选取出来。
            tmp = -torch.log(softmax(ht.transpose().matmul(self.wei_WP))[indexes].squeeze())   # TODO(check) 所有都是加起来吗? , tmp.shape = (n_batch,)
            loss += tmp.sum()

        return loss

    def train_step(self, input_batch):
        """
            train_step，see it in [](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/)
        """

        self.optimizer.zero_grad()
        loss = self._step_loss(input_batch)
        loss.backward()
        optimizer.step()
        
        return loss


    def eval_test(self, test_dataset): #TODO
        """
            output the eval information and store it at best result

        @ test_dataset: 
            type(test_datset) = list
            test_dataset = [item0=[
                                    userid=int, 
                                    imageid=int,
                                    imagefeat=np.array(),
                                    cap_seq=np.array()
                                  ]
                           ]
        """
        self.eval()
        BLEU_1 = 0.
        BLEU_2 = 0.
        BLEU_3 = 0.
        BLEU_4 = 0.
        ROUGE_L = 0.
        num_files = 0
        with torch.no_grad() : 
            for item in test_dataset:
                output = [1] # store the predicted seq_cap, start with <SOS>
                userid, imageid, imagefeat, caq_seq = item
                assert (type(userid) == int)
                assert (type(imageid) == int)
                assert (type(imagefeat) == np.array and imagefeat.shape == (self.n_img_dim) )
                assert (type(caq_seq) == np.array)
                
                max_cap_len = len(caq_seq)
                n_batch = 1
                img_feat = torch.from_numpy(input_batch.imagefeat)  # (n_feat_dim)
                user_embedding = self.wei_user[userid] # (n_cond_dim)
                input_wid = 1               # 1 <SOS> 0 <EOS>

                c0 = torch.randn(self.n_hidden)       # TODO (???变为Parameter吗需要?)
                h0 = img_feat.unsqueeze(0).matmul(self.wei_WI).squeeze()

                self.cond_lstm.eval_start((h0, c0), user_embedding)
                while input_wid != 0 :
                    lstm_input = self.word_emb(input_wid) # FIXME (int or numpy, need experiment) shape = (dim,)
                    h, c = self.cond_lstm.eval_step(lstm_input) 
                    input_wid = h.unsqueeze(0).matmul(self.wei_WP).numpy().argmax()
                    output.append(input_wid)

                # TODO (make use of output and gt and calculate the score)
                from nlp_score import score
                ref = {1: " ".join(map(str, cap_seq.to_list()))}
                hypo = {1: " ".join(map(str, output_seq))}
                score_map = score(ref, hypo)
                BLEU_1 += score_map['Bleu_1']
                BLEU_2 += score_map['Bleu_2']
                BLEU_3 += score_map['Bleu_3']
                BLEU_4 += score_map['Bleu_4']
                ROUGE_L += score_map['ROUGE_L']


        self._best = 'not record' #TODO add it 

        return ('Bleu - 1gram:' + str(BLEU_1/len(test_dataset)) +
                'Bleu - 2gram:' + BLEU_2/len(test_dataset)) + 
                'Bleu - 3gram:'+ BLEU_3/len(test_dataset)) + 
                'Bleu - 4gram:'+ BLEU_4/len(test_dataset)) +
                'Rouge:'+ ROUGE_L/len(test_dataset)
               )
                
    def best_result(self): 
        # return the best eval value to the caller
        return self._best
