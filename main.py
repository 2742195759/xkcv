import sys
if './utils/nlp-metrics' not in sys.path:
    sys.path.append('./utils/nlp-metrics')
sys.path.append('./utils')

from xklib import space
from xkcv_train import normal_train

Debug = True

def get_args():
    args = space()
    args.dataname = 'ava_pcap'
    args.datapath = '../baseline/AVA_PCap/CLEAN_AVA_FULL_AFTER_SUBJECTIVE_CLEANING.json'
    args.pretrained_path = '../baseline/AVA_PCap/mymodul_10_resnet101.pth'
    args.imagedir = './data/ava_dataset/images/'
    args.batchsize = 80
    args.n_cap_len = 100
    args.epochs = 25
    args.device = 'cuda:0'
    args.n_high_dim = 0
    args.n_cap_len = 50
    args.n_word_dim = 500
    args.n_img_dim = 1000
    args.n_user_dim = 5 
    args.n_F = 300
    args.n_hidden = 2000
    args.optimizer_name = 'adam'
    args.optimizer_lr = 0.0003
    args.optimizer_momentum = 0.0
    args.optimizer_weightdecay=0.001
    args.eval_interval = 100
    args.loss_interval = 1
    args.debug = True
    args.save_path = None
    args.load_path = None #'UserCaption.pth'
    return args

if __name__ == '__main__':
    args = get_args()
    model_name = 'User_Caption'
    print ('[MAIN] start train "User_Caption" model')
    model = normal_train(model_name, args, args.save_path, args.load_path)
    print ('[MAIN] end train')
