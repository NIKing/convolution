import numpy as np

from Model import Model

from pynj.loss import CrossEntropyLoss
from pynj.optim import Adam

import utils.IOUtil as IOUtil 

np.set_printoptions(linewidth=180)

model = Model()

optimizer = Adam(model)
loss_fn = CrossEntropyLoss(model)

def train():
    max_epoch = 3

    for i in range(max_epoch):
        
        batch_num = 0 
        current_batch = next(train_data)
        while current_batch:
            predict = model(current_batch['inputs'])

            loss = loss_fn(predict, current_batch['label'])
            
            loss.backward()

            optimizer.step()
            
            batch_num += 1

            print(f'train: {i + 1}/{max_epoch}; {batch_num}/{len(train_data)}; {loss.item()}')


def test():
    current_batch = next(train_data)
    
    results = []
    while current_batch:
        predict = model(current_train, is_train=False)
        results.append(predict, current_batch['label'])
        
    p, r, f1 = evalute(results)
    print(p, r, f1)


def evalute(results):
    pass
    #correct_total = 0
    #for pred, gold in results:
        
        

if __name__ == '__main__':
    
    train_images = IOUtil.read_image_file('./mnist/train-images-idx3-ubyte')
    train_labels = IOUtil.read_label_file('./mnist/train-labels-idx1-ubyte')
    print(train_images[0])
    print(train_images[1])

    #train()

    #test()
    
            
