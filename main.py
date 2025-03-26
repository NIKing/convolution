import random
import numpy as np

from Model import Model
from dataloader import DataLoader

from pynj.loss import CrossEntropyLoss
from pynj.optim import Adam

import utils.IOUtil as IOUtil 

seed=200
random.seed(seed)
np.random.seed(seed)

np.set_printoptions(linewidth=180)

class Framework():
    def __init__(self, model):
        self.model = model

        self.optimizer = Adam(model)
        self.loss_fn = CrossEntropyLoss(model)

    def train(self, train_data):
        max_epoch = 3

        for i in range(max_epoch):
            
            batch_num = 0 
            current_batch = next(train_data)

            while current_batch:

                predict = self.model(current_batch['images'])
                print('预测', predict)

                loss = self.loss_fn(predict, current_batch['labels'])
                
                loss.backward()

                self.optimizer.step()
                
                batch_num += 1

                print(f'train: {i + 1}/{max_epoch}; {batch_num}/{len(train_data)}; {loss.item()}')

                current_batch = next(train_data)


    def test(self, test_data):
        current_batch = next(test_data)
        
        results = []
        while current_batch:
            predict = model(current_train, is_train=False)
            results.append(predict, current_batch['label'])

            current_batch = next(test_data)
            
        p, r, f1 = evalute(results)
        print(p, r, f1)

    def evalute(self, results):
        pass
        #correct_total = 0
        #for pred, gold in results:
        

if __name__ == '__main__':
    
    train_images = IOUtil.read_image_file('./mnist/train-images-idx3-ubyte')
    train_labels = IOUtil.read_label_file('./mnist/train-labels-idx1-ubyte')
    
    train_data = DataLoader(train_images[:100], train_labels[:100])
    model = Model(label_size = train_data.max_label_size)
    
    framework = Framework(model)
    framework.train(train_data)

    #test()
    
            
