############
#  里面可以使用你随意的其他的Dataloader函数。例如torchvision.datasets等
###########

############
#  Mutable XXX
#  注意这个类是很容易变化的，所以可以不要想着去写类重用，可以使用函数重用
#  开始设计就是dataloader根据model变化而变化
###########

###########
#  XXX 数据处理中包含了模型相关和模型无关。:
#  模型相关指的是数据的变化，会影响模型的参数的处理部分。
#  模型无关指的是数据的变化不会影响模型的部分，也就是他们是完全独立的。
#  比如：数据集中，单词的vocab_size是会影响模型的参数的。所以我们要把这部分处理写入 model_related_process中，这部分要尽量简单，否则简化数据集测试不方便。
#        但是一些不会影响的就无所谓，比如cnn处理部分。不同的数据集也是不影响模型的。
###########
import import_path
from torchvision import transforms
from torchvision.models import resnet101
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from xklib import *
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
    dataloader.make_data()
    return dataloader

class dataloader:
    def __init__ (self): #XXX 快速操作，不要有过多的动作，其他的
        """
        """
        pass
    def __len__(self):
        pass
    def shuffle(self):
        pass
    def get_batch(self, batch_id):
        pass
    def get_batch_num(self):
        pass
    def get_testset(self):
        pass
    def make_data(self):
        """
            @ hook func
            Use this function to prepare data, may cost a lot of time
        """
        print ('================Fast Pass=====================')
        self._fast_pass()
        print ('================Slow Pass=====================')
        self._slow_pass()
        self._set_args_()
        pass

    def _set_args_(self):
        """
            @ override
        """
        pass

    @staticmethod
    def _batch_num_cal_(tot, bs):
        a = tot
        b = bs
        assert(isinstance(a, int))
        assert(isinstance(b, int))
        return a // b + (a % b != 0)

    @staticmethod
    def _get_batch_from_list_(l, bid, bs):
        return l[bid*bs:min(bid*bs+bs, len(l))]

    def _fast_pass(self):
        """
            @ override
        """
        pass

    def _slow_pass(self):
        """
            @ override
        """
        pass

class User_Caption(dataloader):
    def __init__(self, args): 
        super(User_Caption, self).__init__()
        import pdb
        pdb.set_trace()
        self.dataname = args.dataname
        self.datapath = args.datapath
        self.imagedir = args.imagedir
        self.batchsize = args.batchsize
        self.pretrained_path = './data/imageweight/resnet101.pth'
        self.device = torch.device(args.device)
        self.n_cap_len = args.n_cap_len
        self.args = args

    def _fast_pass(self):
        self.raw_dataset = json.load(open(self.datapath))
        self.dictionary = None
        self.user2id = Hasher()
        all_docs = []
        user_names = []
        images = self.raw_dataset['images']
        for img in images:
            for sent in img['sentences'] : 
                tokens = [ i for i,t in sent['tokens'] ]
                all_docs.append(tokens)
            for uid in img['uids'] : 
                user_names.append(uid)

        self.user2id.feed(user_names)
        self.n_user = self.user2id.size()
        self.dictionary = Dictionary(all_docs)
        
    def _slow_pass(self):
        cnn = resnet101().to(self.device)
        cnn.load_state_dict(torch.load(self.pretrained_path))
        dataset = self.raw_dataset
        images = dataset['images']

# FIXME
        tmp = images[:10]
        tmp.extend([ i.copy() for i in tmp[-10:]])
        for i in tmp[-10:]:
            i['split'] = 'test'
        #tmp.extend([i for i in images if i['split'] != 'train'][:10])
        images = tmp
# FIXME end #####

        print ('================extract image feature======================')
        new_images = []
        with torch.no_grad():
            for img in tqdm(images):
                img['imgfeat'] = cnn(torch.from_numpy(image_transforms(default_loader(self.imagedir+img['filename'])).unsqueeze(0).numpy()).to(self.device)).to('cpu').numpy().squeeze()
                new_images.append(img)
        images = new_images
        n_voc_size = len(self.dictionary.token2id) #XXX 0 based

        """ uid to batch""" 
        self.dataset = {} #XXX {uid: [item]} item = [img_feat, cap_seq]  cap_seq = [<EOS>,1,2,3,<EOE>]
        self.testset = []
        cnt = 0 
        for img in images:
            for username, sent in zip(img['uids'], img['sentences']) : 
                word_np = np.ones((self.n_cap_len), dtype=np.int32) * (n_voc_size+1)
                userid = self.user2id.tran(username)
                wordid = [ self.dictionary.token2id[i] for i,t in sent['tokens'] ]
                raw_sent = [ i for i, t in sent['tokens'] ]
                wordid.insert(0, n_voc_size)

                """ DEBUG """
                if wordid[1] == 630: 
                    cnt += 1
                print (raw_sent)

                if (len(wordid) >= self.n_cap_len) : 
                    print ('[WARN] delete so long caption')
                    continue 
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

        print ('HMM occurence', cnt)


        self.n_voc_size = n_voc_size + 2

    def __len__(self):
        return self.get_batch_num()

    def get_batch(self, batch_id):
        bs = self.batchsize
        n = batch_id
        batchset = space()
        batchlist = []
        uid = None
        for u, ls in self.dataset.items():
            max_bn = self._batch_num_cal_(len(ls), bs)
            if n >= max_bn: 
                n -= max_bn
                continue
            batchlist = self._get_batch_from_list_(ls, n, bs)
            uid = u
            break

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
        print ("=========== Save Args=============")
        print ('user num:', self.n_user)
        print ('n_voc_size:', self.n_voc_size)
        print ('test num:', len(self.testset))
        print ('train num:', self.get_batch_num())
        self.args.n_user = self.n_user
        self.args.n_voc_size = self.n_voc_size
        self.args.dictionary = self.dictionary

    def get_testset(self):
        return self.testset

if __name__ == '__main__':
    args = fake_space()
    args.device = 'cuda:0'
    dl = User_Caption(args)
    dl.batchsize = 2
    dl.dataset = {0: [[0,1],[0,2],[0,3]], 1:[[0,4]], 2:[[0,5],[0,6]], 3:[[0,7]]}
    print  ("============Test Start=============")
    assert (dl.get_batch_num() == 5)
    assert (dl.get_batch(0).uid == 0)
    assert ((dl.get_batch(0).cap_seq == np.array([1,2],dtype=np.long)).all())
    assert (dl.get_batch(1).uid == 0)
    assert ((dl.get_batch(1).cap_seq == np.array([3],dtype=np.long)).all())
    assert (dl.get_batch(2).uid == 1)
    assert ((dl.get_batch(2).cap_seq == np.array([4],dtype=np.long)).all())
    assert (dl.get_batch(3).uid == 2)
    assert ((dl.get_batch(3).cap_seq == np.array([5,6],dtype=np.long)).all())
    assert (dl.get_batch(4).uid == 3)
    assert ((dl.get_batch(4).cap_seq == np.array([7],dtype=np.long)).all())
    print  ('============xkcv_dataloader tested successfully!!=================')

