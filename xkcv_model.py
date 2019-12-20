## python3 代码
import numpy as np
import torch
import os
import pandas as pd
import xkcv_optimizer
from torch import *
import torch.nn.init as init
from model import *
from nlp_score import score # 评分函数
from xklib import *

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
#3.    XXX(don't let the nn.module have loss) if you need loss, make it can be assessed by the 
#   driver module xvcvmodule, such as self.for_loss_XXX = mid_variates, the self.mid_variates
#   should only contains the information, not the loss itself
#
#      XXX Eval protocal
#1. if have two different state, use the forward() means  See. Cond_LSTM
# 
#      XXX args protocal
#1. The module should only get the values in __init__()
#2. The xkmodule can get args in the args = space() variates
# 
#      XXX input general protocal
#1. every batch appears in the first dim even the batch is 1, the dim will preserved
#      [[item_1]]
#
####################################



dict_path = './cache/xkmodel_save/'

def get_instance(name, args, path=None):
    model = eval(name)(args)
    if (path != None):
        path = dict_path + path
        if os.path.exists(path) :
            model._load_model(path)
    return model

def save_xkmodel(model, path):
    if not os.path.exists(dict_path) : 
        os.mkdir(dict_path)
    model._save_model(dict_path+path)

# XXX (Driver Module, contains a lot nn.module and this is a driven model)
#     1. handle loss and train 
#     2. can use to eval and 

class xkcv_model(torch.nn.Module) :
    def __init__(self):
        super(xkcv_model, self).__init__()
        pass

    def _step_loss(self, input_batch):
        raise NotImplementedError()
        pass

    def train_step(self, input_batch, batch=None, bid=None):
        """
            train_step，see it in [](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/)
        """
        #XXX must add the self.train()
        self.train()

        self.optimizer.zero_grad()
        loss = self._step_loss(input_batch, batch, bid)
        loss.backward()
        self.optimizer.step()

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

    def _save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def _load_model(self, path):
        self.load_state_dict(torch.load(path))

# XXX (需要注意，句子要加入 <SOS> 和 <EOS> 在句首和句尾
class User_Caption(xkcv_model):
    def __init__(self, args):
        super(User_Caption, self).__init__()
        self.dictionary = args.dictionary
        self.batch_size = args.batchsize
        self.device = torch.device(args.device)
        self.n_user = args.n_user
        self.n_user_dim = args.n_user_dim
        self.n_img_dim = args.n_img_dim   # [int] img feature dim
        self.n_word_dim = args.n_word_dim
        self.n_voc_size = args.n_voc_size
        self.n_cap_len  = args.n_cap_len
        self.n_high_dim = args.n_high_dim
        self.n_hidden = args.n_hidden
        self.wei_user = torch.nn.Parameter(Tensor(self.n_user, self.n_user_dim))
        # construct the link to the lstm
        self.wei_WI = torch.nn.Parameter(Tensor(self.n_img_dim+self.n_user_dim, self.n_hidden))   # self.wei_WI 见图, image_feat transform matrix
        self.wei_WC = torch.nn.Parameter(Tensor(self.n_img_dim+self.n_user_dim, self.n_hidden))   # self.wei_WI 见图, image_feat transform matrix
        self.wei_WP = torch.nn.Parameter(Tensor(self.n_hidden, self.n_voc_size))  # self.wei_WP 预测softmax的地方，
        # === submodule ===
        self.cond_lstm = torch.nn.LSTM(self.n_word_dim, self.n_hidden, 1, True, False) # dropout 必须多层LSTM
        self.word_emb = torch.nn.Embedding(self.n_voc_size, self.n_word_dim)
        
        # XXX 初始化变量要reset_parameter中添加
        self.reset_parameter()
        self.optimizer = xkcv_optimizer.get_instance(self, args) # XXX 一定要在 所有parameter之后

        def compare_cos(v_a, v_b):
            import pdb
            pdb.set_trace()
            assert (isinstance(v_a, torch.Tensor))
            assert (isinstance(v_b, torch.Tensor))
            print  ('similarity is : ', torch.nn.CosineSimilarity(2)(v_a, v_b))

        self.compare = DelayArgProcessor(2, compare_cos)

    def reset_parameter(self):
        self.wei_user = init.normal_(self.wei_user, mean=0.0, std=1.0)
        self.wei_WI   = init.normal_(self.wei_WI, mean=0.0, std=10.0)
        #self.wei_WC   = init.normal_(self.wei_WI, mean=0.0, std=1.0)
        self.wei_WP   = init.normal_(self.wei_WP, mean=0.0, std=1.0)

    def loss_function (self, raw_feature, high_feature):
        """ @ mutable
            
            this function is the loss of the high_feature, unsupervised loss. used by the father model
            to calculate loss if needed. if don't need please set the lossfunction to None

            @raw_feature: np.array()  .shape = (self.bs, self.dim)
            @return     : np.array()  .shape = (self.bs, self.fdim)
        """
        # TODO (try different loss function, the basic is to use tag-predict crossentropy like loss)


    def _step_loss(self, input_batch, epochid, batchid): # TODO 将这个拆分为 forward 和 loss_fn 两个
        """
        输入格式: 
        @ input_batch : 
            type(input_batch) = space @后为关键字
            @ key [int]   uid      : 0-base
            @ key [numpy] img_feat : .shape = (n_batch, n_img_dim)
            @ key [numpy] cap_seq  : .shape = (n_batch, n_cap_len)
            ------------------------------------------------------
            <SOS>  voc_size
            <SOE>  voc_size+1
        """
        n_batch = input_batch.img_feat.shape[0]
        user_embedding = self.wei_user[input_batch.uid] # (n_user_dim,)
        raw_img_feat = torch.from_numpy(input_batch.img_feat)
        img_feat = torch.cat([raw_img_feat, user_embedding.repeat(n_batch, 1)], 1)
        lstm_input = torch.transpose(self.word_emb(torch.from_numpy(input_batch.cap_seq)), 0, 1) # n_step, n_batch, dim
        """
            h0.shape = [1, n_batch, n_img_final_dim]
        """
        h0 = img_feat.matmul(self.wei_WI).unsqueeze(0) # FIXME 梯度爆炸了  #shapoe
        # c0 = img_feat.matmul(self.wei_WC).unsqueeze(0) # FIXME 梯度爆炸了
        c0 = torch.full_like(h0, 0)
        output, (h, c) = self.cond_lstm(lstm_input, (h0,c0)) 
        softmax = torch.nn.Softmax(dim=-1)
        loss = torch.tensor(0, dtype=torch.float32) # 初始化为0, 否则
        h = output

        """ {DEBUG} """
        #import pdb
        #pdb.set_trace()
        """
            NOTE:(DEBUG/FIXED)
                这里h[i]是第i个输入得到的结果。所以h[i]应该和cap_seq[i+1]比较结果
        """

        for i, ht in enumerate(h[:-1]): # TODO(check) <EOS>怎么考虑,然后Loss计算要考虑后面的吗。
            indexes = (range(n_batch), input_batch.cap_seq[:,i+1])  # 同一个step中，所有的batch的gt对应的prob选取出来。
            tmp = -torch.log(softmax(ht.matmul(self.wei_WP))[indexes].squeeze())   # TODO(check) 所有都是加起来吗? , tmp.shape = (n_batch,)
            loss += tmp.mean()

        print(h0.shape)
        loss /= len(h[:-1])
        if batchid == 0:
            self.compare.process(h0)
            pass

        return loss * 1.0
#TODO (将id->seq变成一个函数)

    def eval_test(self, test_dataset): 
        """
            output the eval information and store it at best result

        @ test_dataset: 
            type(test_datset) = list
            test_dataset = [item0=[
                                    userid=int, 
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
        num_files = 0.
        once = Once(print)
        #import pdb
        #pdb.set_trace()
        with torch.no_grad() : 
            for item in test_dataset:
                output = [self.n_voc_size-2] # store the predicted seq_cap, start with <SOS>
                userid, imagefeat, cap_seq = item
                assert (type(userid) == int)
                assert (type(imagefeat) == np.ndarray and imagefeat.shape == (self.n_img_dim,) )
                assert (type(cap_seq) == np.ndarray)

                max_cap_len = len(cap_seq)
                n_batch = 1
                raw_img_feat = torch.unsqueeze(torch.from_numpy(imagefeat), 0)
                user_embedding = torch.unsqueeze(self.wei_user[userid], 0)   # (n_cond_dim, )
                imgfeat = torch.cat([raw_img_feat, user_embedding], dim=1)
                input_wid =  self.n_voc_size-2              # size-2 <SOS> size-1 <EOS>
                h0 = imgfeat.matmul(self.wei_WI).unsqueeze(0)
                #c0 = imgfeat.matmul(self.wei_WC).unsqueeze(0)
                c0 = torch.full_like(h0, 0)
                while input_wid != self.n_voc_size-1 and len(output) < 52:
                    lstm_input = self.word_emb(torch.tensor(input_wid)).repeat(1,1,1) # shape = (1,1,dim)
                    _, (h0, c0) = self.cond_lstm(lstm_input, (h0,c0)) 
                    input_wid = h0[0].matmul(self.wei_WP).numpy().argmax()
                    idx = (h0[0].matmul(self.wei_WP).numpy()).squeeze().argsort()
                    ttt = h0[0].matmul(self.wei_WP).numpy().squeeze()[idx]
                    once.process(ttt, idx)
                    output.append(input_wid)

                # TODO (make use of output and gt and calculate the score)
                output_str = " ".join([ self.dictionary[i] for i in output if i < len(self.dictionary.token2id)])
                gt_str = " ".join([ self.dictionary[i] for i in cap_seq.tolist() if i < len(self.dictionary.token2id)])
                print ('pred:', output_str)
                print ('gt  :', gt_str)
                from nlp_score import score
                ref = {1: [gt_str]}
                hypo = {1: [output_str]}
                score_map = score(ref, hypo)
                BLEU_1 += score_map['Bleu_1']
                BLEU_2 += score_map['Bleu_2']
                BLEU_3 += score_map['Bleu_3']
                BLEU_4 += score_map['Bleu_4']
                ROUGE_L += score_map['ROUGE_L']


        self._best = 'not record' #TODO add it 

        return (None , ('\nBleu1:' + str(BLEU_1/len(test_dataset)) + # TODO add the result to this function
                '\nBleu2:' + str(BLEU_2/len(test_dataset)) + 
                '\nBleu3:'+ str(BLEU_3/len(test_dataset)) + 
                '\nBleu4:'+ str(BLEU_4/len(test_dataset)) +
                '\nRouge:'+ str(ROUGE_L/len(test_dataset))
               ))
                
    def best_result(self): 
        # return the best eval value to the caller
        return self._best

