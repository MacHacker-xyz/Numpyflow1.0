import numpy as np
import numpyflow.util
import copy

class Layer:
    def __init__(self):
        self.params = {}
        self.grad = {}
        self.optim_params = {}
        pass

    def forward(self, data, train_flg=True):
        pass

    def backward(self, derivation):
        pass

class Relu(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, data, train_flg= True):
        self.mask = data>0
        return np.maximum(0,data)

    def backward(self, derivation):
        return self.mask*derivation

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, data, train_flg=True):
        self.out = 1 / (1 + np.exp(-data))
        return self.out

    def backward(self, derivation):
        return derivation*self.out*(1-self.out)

class Affine(Layer):
    def __init__(self, in_channels, out_channels, offset = True ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.offset = offset

        # initialize weight and offset
        self.params["w"] = np.random.randn(in_channels, out_channels)/np.sqrt(in_channels)
        if offset:
            self.params["b"] = np.zeros(out_channels)

        # initialize gradient
        self.grad["w"] = np.zeros((in_channels, out_channels))
        if offset:
            self.grad["b"] = np.zeros(out_channels)

        # initialize optim_params
        self.optim_params["w"] = np.zeros((in_channels, out_channels))
        if offset:
            self.optim_params["b"] = np.zeros(out_channels)

        self.x = None

    def forward(self, data, train_flg = True):
        self.x = data
        if self.offset:
            return np.dot(data, self.params["w"])+self.params["b"]
        else:
            return np.dot(data, self.params["w"])

    def backward(self, derivation):
        dx = np.dot(derivation, self.params["w"].T)
        self.grad["w"] = np.dot(self.x.T, derivation)
        if self.offset:
            self.grad["b"] = np.sum(derivation, axis=0)
        return dx

class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.x_exp = None
        self.sum = None
        self.result = None

    
    def forward(self, data, train_flg = True):
        data = data-np.max(data)
        self.x_exp = np.exp(data)
        self.sum = np.sum(self.x_exp,axis = 1)
        self.sum = np.repeat(self.sum, data.shape[1])
        self.sum.resize(data.shape)
        self.result = self.x_exp/self.sum
        return self.result

    def backward(self, derivation):
        temp = self.x_exp * derivation
        temp = np.sum(temp, axis=1)
        temp = np.repeat(temp, self.sum.shape[1])
        temp.resize(self.sum.shape)

        return self.result*derivation-self.x_exp*temp/(self.sum**2)
        
class Dropout(Layer):
    def __init__(self, dropout_ratio = 0.5):
        super().__init__()
        self.mask = None
        self.dropout_ratio = dropout_ratio
    
    def forward(self, data, train_flg = True):
        if train_flg:
            self.mask = np.random.rand(*data.shape) > self.dropout_ratio
            return data * self.mask
        else:
            return data * (1.0 - self.dropout_ratio)
    
    def backward(self, derivation):
        return derivation * self.mask

class BatchNormalization(Layer):
    """这段做正则归一化的代码是抄的，嘻嘻！
    已经对numpyflow做了适配
    """    
    def __init__(self, gamma = 1, beta =0, momentum=0.9, running_mean=None, running_var=None):
        super().__init__()

        self.params["gamma"] = gamma
        self.params["beta"] = beta

        self.grad["gamma"] = None
        self.grad["beta"] = None

        self.optim_params["gamma"] = 0
        self.optim_params["beta"] = 0

        self.momentum = momentum
        self.input_shape = None # Conv层的情况下为4维，全连接层的情况下为2维  

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
    
    def forward(self, data, train_flg=True):
        x = data
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.params["gamma"] * xn + self.params["beta"]
        return out

    def backward(self, derivation):
        if derivation.ndim != 2:
            N, C, H, W = derivation.shape
            derivation = derivation.reshape(N, -1)

        dx = self.__backward(derivation)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, derivation):
        dbeta = derivation.sum(axis=0)
        dgamma = np.sum(self.xn * derivation, axis=0)
        dxn = self.params["gamma"] * derivation
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.grad["gamma"] = dgamma
        self.grad["beta"] = dbeta
        
        return dx



class MSELoss(Layer):
    def __init__(self):
        super().__init__()
        self.label = None
        self.in_ = None
    
    def forward(self, data, label):
        self.in_ = data
        self.label = label
        data = 0.5*(data - label)**2
        return np.sum(data)/data.size

    def backward(self):
        return self.in_-self.label


class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()
        self.label = None
        self.in_ = None
    
    def forward(self, data, label):
        self.in_ = data
        self.label = label
        data = -self.label*np.log(self.in_)
        return np.sum(data)/data.size

    def backward(self):
        return -self.label/self.in_



class Convolution(Layer):
    def __init__(self, in_channels, out_channels,kernel_size = 3, stride=1, padding=0, bias = True):
        super().__init__()
        # initialize some params
        self.params["w"] = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)/np.sqrt(in_channels*kernel_size**2)
        if bias:
            self.params["b"] = np.zeros(out_channels)

        # initialize the grads
        self.grad["w"] = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.grad["b"] = np.zeros(out_channels)
        
        # initialize the optim_params
        self.optim_params["w"] = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.optim_params["b"] = np.zeros(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None
        

    def forward(self, x, train_flg = True):
        FN, C, FH, FW = self.params["w"].shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.padding - FH) / self.stride)
        out_w = 1 + int((W + 2*self.padding - FW) / self.stride)

        col = numpyflow.util.im2col(x, FH, FW, self.stride, self.padding)
        col_W = self.params["w"].reshape(FN, -1).T

        if self.bias:
            out = np.dot(col, col_W) + self.params["b"]
        else:
            out = np.dot(col, col_W)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, derivation):
        FN, C, FH, FW = self.params["w"].shape
        derivation = derivation.transpose(0,2,3,1).reshape(-1, FN)
        if self.bias:
            self.grad["b"] = np.sum(derivation, axis=0)
        self.grad["w"] = np.dot(self.col.T, derivation)
        self.grad["w"] = self.grad["w"].transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(derivation, self.col_W.T)
        dx = numpyflow.util.col2im(dcol, self.x.shape, FH, FW, self.stride, self.padding)

        return dx



class MaxPooling(Layer):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.x = None
        self.arg_max = None

    def forward(self, data, train_flg = True):
        N, C, H, W = data.shape
        out_h = int(1 + (H - self.kernel_size) / self.stride)
        out_w = int(1 + (W - self.kernel_size) / self.stride)

        col = numpyflow.util.im2col(data, self.kernel_size, self.kernel_size, self.stride, self.padding)
        col = col.reshape(-1, self.kernel_size*self.kernel_size)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = data
        self.arg_max = arg_max

        return out

    def backward(self, derivation):
        derivation = derivation.transpose(0, 2, 3, 1)
        
        pool_size = self.kernel_size * self.kernel_size
        dmax = np.zeros((derivation.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = derivation.flatten()
        dmax = dmax.reshape(derivation.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = numpyflow.util.col2im(dcol, self.x.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)
        
        return dx


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.shape = None
    
    def forward(self, data, train_flg = True):
        self.shape = data.shape
        return data.reshape(data.shape[0],-1)

    def backward(self, derivation):
        return derivation.reshape(self.shape)

if __name__=="__main__":
    pass