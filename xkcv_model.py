## python3 代码
import numpy as np
import torch
import os
import pandas as pd
import xkcv_optimizer
from torch import *
import torch.nn.init as init
from nlp_score import score # 评分函数

####################################
# 
#      XXX xkcv_model protocol
#1. compared with nn.Module, the xkcv_model have the abillity to 
#   handle the loss. because a successful model contains 2 parts
#   the model parts(forward parts) and the loss parts
#
#      XXX Loss process protocol (always xkcv)
#1. Loss contains 2 kinds of : supervised and unsupervised
#2. for the supervised learning and inter_module unsupervised loss, the loss is calculated
#   by the father xkcv module, and in the function of the _step_loss() function
#3. XXX don't let the nn.module have loss, if you need them, make it can be assessed by the 
#   driver module xvcvmodule, such as self.for_loss_XXX = mid_variates
#
#      XXX Eval protocal
#1. if have two different state, use the forward() means  See. Cond_LSTM
#
####################################



dict_path = './xkmodel_param/'

def get_instance(name, args, path=None):
    model = eval(name)(args)
    if (path != None):
        path = dict_path + path
        model.load_state_dict(torch.load(path))
    return model

def save_xkmodel(model, path):
    if not os.path.exists(dict_path) : 
        os.mkdir(dict_path)
    torch.save(model.state_dict(), dict_path+path)

# XXX (Driver Module, contains a lot nn.module and this is a driven model)
#     1. handle loss and train 
#     2. can use to eval and 

class xkcv_model(nn.Module) :
    def __init__(self):
        super(xkcv_model, self).__init__()
        pass

    def _step_loss(self, input_batch):
        raise NotImplementedError()
        pass

    def train_step(self, input_batch):
        """
            train_step，see it in [](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/)
        """
        #XXX must add the self.train()
        self.train()

        self.optimizer.zero_grad()
        loss = self._step_loss(input_batch)
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def eval_test(self, test_dataset): 
        """
            the forward process and the result
            @test_dataset : should be a list of input, input formated see the concrete function
            @return       : should be a tuple of ([result_format], avg_score/avg_metric_str)
        """
        #XXX must add self.eval()
        raise NotImplementedError()
        
    def best_result(self):  
        raise NotImplementedError()


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

        return (None , ('Bleu - 1gram:' + str(BLEU_1/len(test_dataset)) + # TODO add the result to this function
                'Bleu - 2gram:' + str(BLEU_2/len(test_dataset)) + 
                'Bleu - 3gram:'+ str(BLEU_3/len(test_dataset)) + 
                'Bleu - 4gram:'+ str(BLEU_4/len(test_dataset)) +
                'Rouge:'+ str(ROUGE_L/len(test_dataset))
               ))
                
    def best_result(self): 
        # return the best eval value to the caller
        return self._best



class 
