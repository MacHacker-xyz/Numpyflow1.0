# Numpyflow1.0 使用numpy搭建的简单的深度学习框架

# Numpyflow1.0 Framework for deep learning (An implementation of Pytorch-like framework in Numpy)

此框架参考了pytorch的一些实现方式，使用上和pytorch相似，特点上可以利用单一的命令实现网络梯度的反传，以及网络的参数更新

此框架完全用numpy库完成，实现了一些深度学习的基本的算法，包括conv2d, MaxPooling, Linear, Softmax, Relu, Sigmoid, MSELoss, CrossEntropyLoss, 以及一些基本的神经网络操作

其中包括模块dataset, 其中封装了几种主流的数据库，包括mnist, cifar10, tiny-image-net, 可以直接调用

在numpyflow中包括了Layer, Model, Optimizer, util四个模块，其中Layer中封装了conv2d, MaxPooling, Linear, Softmax, Relu, Sigmoid, MSELoss, CrossEntropyLoss, 以及一些基本的神经网络操作。
Model为神经网络的父类。Optimizer中封装了几种典型的优化器，包括SGD, Momentum, AdaGard可供选择。

本例中提供了一个案例，使用该框架初步实现了包含残差块的网络，网络结构如下：
<img width="973" alt="截屏2021-08-31 上午9 28 18" src="https://user-images.githubusercontent.com/77945509/131426764-41c9d598-9519-47df-92b3-8e8c17a3dcb0.png">

> 本框架参考了《深度学习入门--基于Python的理论与实践》中的源代码，实现了其中的全部功能，并将正穿，反传，以及参数更新进行了封装，实现了一部到位，提升了框架的易用性

邮箱：hzb21@mails.tsinghua.edu.cn

