import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PatchDataTrain(Dataset):

    def __init__(self, source='32_sample'):
        self.x = []
        self.y = []
        self.classification = []
        data_path = 'data/' + source + '/train'
        for classification in os.listdir(data_path):
            self.classification.append(classification)
            for sample in os.listdir(os.path.join(data_path, classification)):
                self.x.append(os.path.join(data_path, classification, sample))
                self.y.append(len(self.classification)-1)  # 类别在self.classification列表中的位置作为标签

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = torch.Tensor(np.expand_dims(np.array(Image.open(self.x[index])), axis=0))
        y = torch.LongTensor(self.y)[index]
        return x.cuda(), y.cuda()

    def getlabel(self):
        return self.classification


class PatchDataTest(Dataset):

    def __init__(self, source='32_sample'):
        self.x = []
        self.y = []
        self.classification = []
        data_path = 'data/' + source + '/test'
        for classification in os.listdir(data_path):
            self.classification.append(classification)
            for sample in os.listdir(os.path.join(data_path, classification)):
                self.x.append(os.path.join(data_path, classification, sample))
                self.y.append(len(self.classification) - 1)  # 类别在self.classification列表中的位置作为标签

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = torch.Tensor(np.expand_dims(np.array(Image.open(self.x[index])), axis=0))
        y = torch.LongTensor(self.y)[index]
        return x.cuda(), y.cuda()

    def getlabel(self):
        return self.classification
