############
#  里面可以使用你随意的其他的Dataloader函数。例如torchvision.datasets等
###########

############
#  Mutable
#  注意这个类是很容易变化的，所以可以不要想着去写类重用，可以使用函数重用
#  开始设计就是dataloader根据model变化而变化
###########
from torchvision import transforms
from torchvision.models import resnet101
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from xklib import space, Hasher
import torch
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
# copy from the loader function
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        image = Image.open(path)
        image = image.convert('RGB')
        return image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

image_transforms = transforms.Compose([
    transforms.Resize((224,224)),
#    transforms.RandomSizedCrop(600),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

"""=============================================================================="""

dict_path = './cache/xkdataloader_save/'

def get_instance(name, args):
    dataloader = eval(name)(args)
    return dataloader

class dataloader:
    def __init__ (self):
        pass
    def shuffle(self):
        pass
    def __len__(self):
        pass
    def get_batch(self, batch_id):
        pass
    def get_batch_num(self):
        pass
    def get_testset(self):
        pass
    def _set_args_(self):
        pass
    def _batch_num_cal_(self, tot, bs):
        a = tot
        b = bs
        assert(isinstance(a, int))
        assert(isinstance(b, int))
        return a // b + (a % b != 0)

    def _get_batch_from_list_(self, l, bid, bs):
        return l[bid*bs:min(bid*bs+bs, len(l))]

class User_Caption(dataloader):
    def __init__(self, args): 
        super(User_Caption, self).__init__()
        self.dataname = args.dataname
        self.datapath = args.datapath
        self.imagedir = args.imagedir
        self.batchsize = args.batchsize
        self.pretrained_path = './data/imageweight/resnet101.pth'
        self.device = torch.device(args.device)
        self.n_cap_len = args.n_cap_len
        self.args = args
        self._loaddata_()
        self._set_args_()

    def _loaddata_(self):
        cnn = resnet101().to(self.device)
        cnn.load_state_dict(torch.load(self.pretrained_path))
        dataset = json.load(open(self.datapath))
        images = dataset['images']
#FIXME
        tmp = images[:100]
        tmp.extend([i for i in images if i['split'] != 'train'][:10])
        images = tmp
#FIXME end #####


        print ('================extract image feature======================')
        new_images = []
        with torch.no_grad():
            for img in tqdm(images):
                img['imgfeat'] = cnn(torch.from_numpy(image_transforms(default_loader(self.imagedir+img['filename'])).unsqueeze(0).numpy()).to(self.device)).to('cpu').numpy().squeeze()
                new_images.append(img)
        images = new_images
        print ('================end and save new json =====================')
        self.dictionary = None
        self.user2id = Hasher()
        all_docs = []
        user_names = []
        for img in images:
            for sent in img['sentences'] : 
                tokens = [ i for i,t in sent['tokens'] ]
                all_docs.append(tokens)
            for uid in img['uids'] : 
                user_names.append(uid)

        self.user2id.feed(user_names)
        self.n_user = self.user2id.size()
        self.dictionary = Dictionary(all_docs)
        n_voc_size = len(self.dictionary.token2id) #XXX 0/1 based?
        """ uid to batch""" 

        self.dataset = {} 
        self.testset = []
        for img in images:
            for username, sent in zip(img['uids'], img['sentences']) : 
                word_np = np.ones((self.n_cap_len), dtype=np.int32) * (n_voc_size+1)
                userid = self.user2id.tran(username)
                wordid = [ self.dictionary.token2id[i] for i,t in sent['tokens'] ]
                wordid.insert(0, n_voc_size)
                assert(len(wordid) < self.n_cap_len)
                word_np[0:len(wordid)] = np.array(wordid, dtype=np.int32)
                if img['split'] == 'train' :
                    tmp = self.dataset.get(userid, [])
                    tmp.append([img['imgfeat'], word_np])
                    self.dataset[userid] = tmp
                else :
                    tmp = []
                    tmp.append(userid)
                    tmp.append(img['imgfeat'])
                    tmp.append(word_np)
                    self.testset.append(tmp)

        self.n_voc_size = n_voc_size + 2
        print ('user num:', self.n_user)
        print ('n_voc_size:', self.n_voc_size)
        print ('test num:', len(self.testset))
        print ('train num:', self.get_batch_num())
        self._set_args_()

    def __len__(self):
        return self.get_batch_num()

    def get_batch(self, batch_id):
        bs = self.batchsize
        n = batch_id
        batchset = space()
        batchlist = []
        uid = None
        for u, ls in self.dataset.items():
            if n >= len(ls) : 
                n -= len(ls)
                continue
            batchlist = self._get_batch_from_list_(ls, n, bs)
            uid = u

        batchset.uid = uid
        batchset.img_feat = np.array([ i for i,j in batchlist ], dtype=np.float32)
        batchset.cap_seq = np.array([ j for i,j in batchlist ], dtype=np.long)
        return batchset
           
    def get_batch_num(self):
        num = 0
        bs = self.batchsize
        for u, ls in self.dataset.items():
            num += self._batch_num_cal_(len(ls), bs)
        return num

    def _set_args_(self):
        self.args.n_user = self.n_user
        self.args.n_voc_size = self.n_voc_size
        self.args.dictionary = self.dictionary

    def get_testset(self):
        return self.testset
