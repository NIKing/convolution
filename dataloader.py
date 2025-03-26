import random
import numpy as np

class DataLoader():
    def __init__(self, images_dataset, labels_dataset, batch_size=1):
        self.batch_size = batch_size
        
        assert len(images_dataset) == len(labels_dataset), '图片数量与标签数量不一致'

        split_size = len(images_dataset) / batch_size
        self.images_dataset = np.array_split(images_dataset, split_size)
        
        self.max_label_size = max(labels_dataset) + 1

        label_ids = self._convert_labels_ids(labels_dataset) 
        self.labels_dataset = np.array_split(label_ids, split_size)

        self.current_index = -1
        self.max_total = len(self.images_dataset)

    def __len__(self):
        return len(self.images_dataset)

    def __getitem__(self, index):
        return {'images': self.images_dataset[index], 'labels': self.labels_dataset[index]}

    def __iter__(self):
        return self 

    def __next__(self):
        self.current_index += 1
        if self.current_index >= self.max_total:
            return {} 
        
        #return np.array([])
        return {'images': self.images_dataset[self.current_index], 'labels': self.labels_dataset[self.current_index]}

    def is_next(self):
        return self.current_index < self.max_total 

    def _convert_labels_ids(self, labels_dataset):
        """转换标签为编号序列""" 
        label_ids = np.zeros((len(labels_dataset), self.max_label_size), dtype=np.int32)

        for i in labels_dataset:
            label_ids[i, labels_dataset[i]] = 1

        return label_ids



    
            
