import copy
from typing import OrderedDict
import numpy as np
import numpyflow.Layer as Layer
import numpyflow.Model

class MyNet(numpyflow.Model.Net):
    def __init__(self):
        super().__init__()
        self.params["Covn1"] = Layer.Convolution(in_channels=3, out_channels= 10, kernel_size=3, stride=1, padding=1, bias=True)
        self.params["BatchNorm1"] = Layer.BatchNormalization()
        self.params["Relu1"] = Layer.Relu()

        self.params["Covn2"] = Layer.Convolution(in_channels=10, out_channels= 10, kernel_size=3, stride=1, padding=1, bias=True)
        self.params["BatchNorm2"] = Layer.BatchNormalization()

        self.params["Covn_res"] = Layer.Convolution(in_channels=3, out_channels=10, kernel_size=3,stride=1,padding=1, bias=True)
        self.params["BatchNorm_res"] = Layer.BatchNormalization()
        
        self.params["Relu2"] = Layer.Relu()
        self.params["MaxPool2"] = Layer.MaxPooling(kernel_size=2, stride=2, padding=0)
        self.params["Flatten2"] = Layer.Flatten()
        self.params["Affine3"] = Layer.Affine(in_channels=10*16*16,out_channels=40,offset=True)
        self.params["BatchNorm3"] = Layer.BatchNormalization()
        self.params["Dropout3"] = Layer.Dropout(dropout_ratio=0.5)
        self.params["Relu3"] = Layer.Relu()
        self.params["Affine4"] = Layer.Affine(in_channels=40,out_channels=20,offset=True)
        self.params["BatchNorm4"] = Layer.BatchNormalization()
        self.params["Dropout4"] = Layer.Dropout(dropout_ratio=0.5)
        self.params["Relu4"] = Layer.Relu()
        self.params["Affine5"] = Layer.Affine(in_channels=20,out_channels=10,offset=True)
        self.params["BatchNorm5"] = Layer.BatchNormalization()
        self.params["Softmax5"] = Layer.Softmax()
        
        self.Loss_fn = Layer.CrossEntropyLoss()

    def forward(self, data, label):

        data_copy = copy.deepcopy(data)

        data = self.params["Covn1"].forward(data, train_flg = self.train_flg)
        data = self.params["BatchNorm1"].forward(data, train_flg = self.train_flg)
        data = self.params["Relu1"].forward(data, train_flg = self.train_flg)
        data = self.params["Covn2"].forward(data, train_flg = self.train_flg)
        data = self.params["BatchNorm2"].forward(data, train_flg = self.train_flg)
        
        data_copy = self.params["Covn_res"].forward(data_copy, train_flg = self.train_flg)
        data_copy = self.params["BatchNorm_res"].forward(data_copy, train_flg = self.train_flg)
        
        data = data+data_copy

        data = self.params["Relu2"].forward(data, train_flg = self.train_flg)
        data = self.params["MaxPool2"].forward(data, train_flg = self.train_flg)
        data = self.params["Flatten2"].forward(data, train_flg = self.train_flg)
        data = self.params["Affine3"].forward(data, train_flg = self.train_flg)
        data = self.params["BatchNorm3"].forward(data, train_flg = self.train_flg)
        data = self.params["Dropout3"].forward(data, train_flg = self.train_flg)
        data = self.params["Relu3"].forward(data, train_flg = self.train_flg)
        data = self.params["Affine4"].forward(data, train_flg = self.train_flg)
        data = self.params["BatchNorm4"].forward(data, train_flg = self.train_flg)
        data = self.params["Dropout4"].forward(data, train_flg = self.train_flg)
        data = self.params["Relu4"].forward(data, train_flg = self.train_flg)
        data = self.params["Affine5"].forward(data, train_flg = self.train_flg)
        data = self.params["BatchNorm5"].forward(data, train_flg = self.train_flg)
        data = self.params["Softmax5"].forward(data, train_flg = self.train_flg)
        
        self.loss = self.Loss_fn.forward(data, label)
        
        self.output = data
        return self.loss

    def backward(self):
        derivation = self.Loss_fn.backward()

        derivation = self.params["Softmax5"].backward(derivation)
        derivation = self.params["BatchNorm5"].backward(derivation)
        derivation = self.params["Affine5"].backward(derivation)
        derivation = self.params["Relu4"].backward(derivation)
        derivation = self.params["Dropout4"].backward(derivation)
        derivation = self.params["BatchNorm4"].backward(derivation)
        derivation = self.params["Affine4"].backward(derivation)
        derivation = self.params["Relu3"].backward(derivation)
        derivation = self.params["Dropout3"].backward(derivation)
        derivation = self.params["BatchNorm3"].backward(derivation)
        derivation = self.params["Affine3"].backward(derivation)
        derivation = self.params["Flatten2"].backward(derivation)
        derivation = self.params["MaxPool2"].backward(derivation)
        derivation = self.params["Relu2"].backward(derivation)

        derivation_copy = copy.deepcopy(derivation)
        derivation = self.params["BatchNorm2"].backward(derivation)
        derivation = self.params["Covn2"].backward(derivation)
        derivation = self.params["Relu1"].backward(derivation)
        derivation = self.params["BatchNorm1"].backward(derivation)
        derivation = self.params["Covn1"].backward(derivation)

        derivation_copy = self.params["BatchNorm_res"].backward(derivation_copy)
        derivation_copy = self.params["Covn_res"].backward(derivation_copy)

if __name__=="__main__":
    pass

