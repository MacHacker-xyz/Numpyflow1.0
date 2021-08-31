# 这个框架中间的optimizer提供了三种方法，且每个optimizer中封装了正向传播和反向传播两种方法
import numpy as np
import copy

class Optimizer:
    def __init__(self, model):
        pass

    def backward(self):
        pass

    def update(self):
        pass

class SDG(Optimizer):
    def __init__(self, model, lr):
        super().__init__(model)
        self.lr = lr
        self.model = model
    
    def update(self):
        for layer in self.model.params.values():
            for key in layer.params.keys():
                layer.params[key] = layer.params[key] - layer.grad[key]*self.lr




class Momentum(Optimizer):
    def __init__(self, model, momentum, lr):
        super().__init__(model)
        self.lr = lr
        self.momentum = momentum
        self.model = model
        
    def update(self):
        for layer in self.model.params.values():
            for key in layer.params.keys():
                layer.optim_params[key] = self.momentum*layer.optim_params[key]  - self.lr*layer.grad[key]
                layer.params[key] = layer.params[key] + layer.optim_params[key]



class AdaGrad(Optimizer):
    def __init__(self, model, lr):
        super().__init__(model)
        self.lr = lr
        self.model = model

    def update(self):
        for layer in self.model.params.values():
            for key in layer.params.keys():
                layer.optim_params[key] = layer.optim_params[key] + np.sum(layer.grad[key]**2)
                layer.params[key] = layer.params[key] - self.lr*layer.grad[key]/(np.sqrt(layer.optim_params[key])+0.0000001)

