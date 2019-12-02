class space (object) : 
    """simple test the namespace of the python
    """

    def __init__ (self) : 
        self.namespace = {}

    def __setitem__(self , key , value) : 
        self.namespace[key] = value

    def __getitem__(self , key) : 
        return self.namespace[key]


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

