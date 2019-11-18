from xklib import space
from xkcv_train import normal_train

def get_args():
    args = space()
# XXX
    pass

if __file__ == '__main__':
    args = get_args()
    model_name = 'User_Caption'
    print ('[MAIN] start train "User_Caption" model')
    model = normal_train(model_name, args)
    print ('[MAIN] end train')
