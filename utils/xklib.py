class space (object) : 
    """simple test the namespace of the python
    """

    def __init__ (self) : 
        self.namespace = {}

    def __setitem__(self , key , value) : 
        self.namespace[key] = value

    def __getattr__(self , key) : 
        return self.namespace[key]

class fake_space(object):
    """
        return every thing as None
    """
    def __init__ (self) : 
        self.namespace = {}

    def __setitem__(self,  key , value) :
        self.namespace[key] = value

    def __getattr__(self, key):
        return self.namespace[key] if key in self.namespace else None


class Hasher(object) : 
    def __init__(self , li=None) : 
        self.tr = {}
        self.inv = {}
        if li != None : 
            self.feed(li)

    def feed(self , li) : 
        assert( isinstance(li, list) )
        cnt = 0
        for name in li : 
            if name not in self.tr : 
                self.tr[name] = cnt 
                self.inv[cnt] = name
                cnt += 1

    def tran(self , name) : 
        return self.tr[name]

    def invt(self , idx) : 
        return self.inv[idx]

    def size(self):
        assert(len(self.tr) == len(self.inv))
        return len(self.tr)

    def testcase() : 
        h = Hasher(['name' , 'xk' , 'wt' , 'xk'])
        assert(h.tran('xk') == 1)
        assert(h.tran('name') == 0)
        assert(h.tran('wt') == 2)
        assert(h.invt(2)=='wt')

##################################
#
#
##################################
class DelayArgProcessor:
    """
       这个函数用来延迟处理参数输入和输出，参数一个一个传递，只要搜集了n个参数，就开始调用process函数
       process 函数包括两步，第一个是检测参数，
    """
    def __init__(self, n, process_func, post_process=None, assert_func=None):
        self.n = n 
        self.args = []
        self._func_ = process_func
        self._post_ = post_process
        self._check_ = assert_func

    def process(self, input_args) : 
        self.args.extend(input_args)
        if len(self.args) == self.n : 
            if self._check_ :
                try : 
                    res = [ self._check_(i) for i in self.args ]
                    assert (sum(res) == 0)
                except AssertionError as e : 
                    print ('[ERROR] input args not compatiable')

            self._func_(*self.args)
            self.args = self._post_(self.args) if self._post_ else []

def tensor_compare_delay_factory():
    def func(a, b):
        print('[Compare]')
        print('\t', a)
        print('\t', b)
    return DelayArgProcessor(2, func)

class Once:
    def __init__(self, func):
        self.init = True
        self.func = func

    def process(self, *args):
        if self.init:
            self.func(*args)
        self.init = False

