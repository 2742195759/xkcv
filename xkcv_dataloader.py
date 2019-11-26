############
#  里面可以使用你随意的其他的Dataloader函数。例如torchvision.datasets等
###########

############
#  Mutable
#  注意这个类是很容易变化的，所以可以不要想着去写类重用，可以使用函数重用
#  开始设计就是dataloader根据model变化而变化
###########

dict_path = './cache/xkdataloader_save/'

def get_instance(name, args, path=None):
    model = eval(name)(args)
    if (path != None):
        path = dict_path + path
        model.load_state_dict(torch.load(path))
    return model

class dataloader:
    def __init__ (self):
        pass
    def shuffle(self):
        pass
    def __len__(self):
        pass
    def get_batch(self, batch_id):
        pass

class User_Caption(dataloader):
    def __init__(self, args): 
        super(UserCaptionDataloader, self).__init__()
    def __len__(self):
        pass
    def get_batch(self, batch_id):
        pass
