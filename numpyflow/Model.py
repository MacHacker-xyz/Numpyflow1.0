from typing import OrderedDict
import copy

class Net():
    def __init__(self):
        self.params = OrderedDict()
        self.Loss_fn = None
        self.output = None
        self.loss = None
        self.train_flg = True
    
    def forward(self, data, label):
        for key,layer in self.params.items():
            data = layer.forward(data, train_flg = self.train_flg)
        self.loss = self.Loss_fn.forward(data, label)
        self.output = data
        return self.loss
    
    def backward(self):
        derivation = self.model.Loss_fn.backward()
        #这里产生一个反向的迭代器，采用了浅层的拷贝，使得神经网络的层仍然指向元神经网络的层
        params = copy.copy(self.params)
        while(len(params)!=0):
            key,layer = params.popitem()
            derivation = layer.backward(derivation)