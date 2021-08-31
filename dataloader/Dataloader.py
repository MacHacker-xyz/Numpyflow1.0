import numpy as np
class DataLoader():
    """This is a dataloader class
    """    
    def __init__(self, data, label, batch_size, epoch_size, is_rand = False):
        """This is a initialization of the class

        Args:
            data : the train or test data
            label : the right labels
            batch_size : the batch size
            epoch_size : the training epoch number
            is_rand (bool, optional): whether to loader data randomly. Defaults to False.
        """        
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.is_rand = is_rand
    
    def __iter__(self):
        if self.is_rand:
            for i in range(0,self.data.shape[0],self.batch_size):
                mask = np.random.randint(0, self.data.shape[0], self.batch_size)
                data = self.data[mask]
                label = self.label[mask]
                yield data,label
        else:
            for i in range(0,self.data.shape[0],self.batch_size):
                data = self.data[i:i+self.batch_size]
                label = self.label[i:i+self.batch_size]
                yield data,label
