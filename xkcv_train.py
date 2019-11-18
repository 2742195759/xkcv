##############################
#   普通驱动函数
#      Main()
##############################

##############################
#
#       Assumption 假设
# 1. 驱动函数中只有numpy 和 pandas 对象, 不包含具体的tensor数据
# 2. 驱动函数通过获取实例来获得model和dataloader
# 3. 驱动函数不使用学习率等，具体的训练过程在model函数中
# 4. 本普通驱动函数不考虑多重训练啥的，只负责简单的训练方法
#
##############################

##############################
#
#      args对象必须参数
#     
#1. 优化器相关
#   optimizer_name 
#   optimizer_lr 
#   optimizer_momentum
# 
#2. 训练相关
#   device  
#   epochs
#   echo_interval 
#   batch_size 
#   eval_interval
#   loss_interval
#
##############################

import os
import sys
import xkcv_model
from xklib import space 
if './utils/nlp-metrics' not in sys.path:
    sys.path.append('./utils/nlp-metrics')

def interface_test(model, dataset, args) : 
    assert(isinstance(args, space))

def normal_train(model_name, args, save=None, load=None):
    model = xkcv_model.get_instance(model_name, args, load)
    dataset = xkcv_dataloader.get_instance(model_name, args)

    interface_test(model, dataset, args)
    for epoch in range(args.epochs):
        dataset.shuffle()                         # XXX 每个epoch调用一次，shuffle
        tot = len(dataset)
        steps = tot / args.batch_size + tot + 1   # XXX dataset 要在 batch_id 过大时返回空
        for bid in range(steps):
            batch = dataset.get_batch(bid)        # XXX dataset.get_batch()  应该返回一个 dict{str: numpy} , 列为对应的名词
            loss  = model.train_step(batch)       # XXX type is float
            if ((bid+1) % args.eval_interval == 0): 
                print ('[epoch:{epoch}, step:{bid}] eval = {eval_str}'.format(epoch=epoch, bid=bid, eval_str=model.eval_test()))
            if ((bid+1) % args.loss_interval == 0): 
                print ('[epoch:{epoch}, step:{bid}] loss = {loss_str}'.format(epoch=epoch, bid=bid, str(loss)))

        print ('[epoch:{epoch}, step:{bid}] {eval_str}'.format(epoch=epoch, bid=bid, eval_str=model.eval_test()))
        print ('[BEST epoch:{epoch}] {eval_str}'.format(epoch=epoch, bid=bid, eval_str=model.best_result()))

    if save :
        xkcv_model.save_xkmodel(model, save)

    return model
