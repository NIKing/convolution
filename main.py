


if __name__ == '__main__':
    
    train_images = util.load('./mnist/train-images-idx3-ubtye.zip')
    train_labels = util.load('./mnist/train-labels-idx1-ubyte.zip')

    
    for i in range(3):
        
         current_train = next(train_data)

         while current_train:
            
            model(current_train)

            
