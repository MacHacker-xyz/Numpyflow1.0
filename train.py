import  numpyflow.Optimizer as Optimizer
import numpy as np
import model
import dataset.dataset as dataset
from dataloader.Dataloader import DataLoader
from tqdm import tqdm

##set some super parameters
Lr = 0.0001
Batch_size = 50
Epoch_size = 400
Is_rand = False

network = model.MyNet()


train_data, train_label, test_data, test_label = dataset.load_CIFAR10()

train_dataloader = DataLoader(
    data = train_data,
    label = train_label,
    batch_size = Batch_size,
    epoch_size = Epoch_size,
    is_rand = Is_rand
)

optimizer = Optimizer.SDG(network, lr=Lr)




for i in range(Epoch_size):
    network.train_flg = True
    loss = []
    pbar = tqdm(total=int(train_data.shape[0]/Batch_size),desc='epoch:{:>3d}/{}'.format(i+1, Epoch_size), leave=False, ncols=100, unit='batch', unit_scale=True)
    for data,label in iter(train_dataloader):
        loss.append(network.forward(data,label))
        network.backward()
        optimizer.update()
        pbar.update(1)
    pbar.close()
    #print the accuracy
    network.train_flg = False
    _ = network.forward(data=test_data,label=test_label)
    test_result = network.output
    test_loss = network.loss
    result_mask = np.argmax(test_result, axis=1)
    label_mask = np.argmax(test_label, axis=1)
    accuracy = float((np.sum(result_mask==label_mask)/result_mask.size)*100)
    print("epoch:{:>3d}/{}  |  loss:{:.8f}  |  accuracy:{:.2f}%".format(i+1,Epoch_size,sum(loss)/len(loss),accuracy))
