import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PatchDataTrain(Dataset):

    def __init__(self, source=('normal_160', 'sobel_32', 'canny_32')):
        self.x = []
        self.y = []
        self.classification = []
        data_path = ('data/' + source[0] + '/train',
                     'data/' + source[1] + '/train',
                     'data/' + source[2] + '/train')
        for classification in os.listdir(data_path[0]):
            self.classification.append(classification)
            for sample in os.listdir(os.path.join(data_path[0], classification)):
                self.x.append((os.path.join(data_path[0], classification, sample),
                               os.path.join(data_path[1], classification, sample),
                               os.path.join(data_path[2], classification, sample)))
                self.y.append(len(self.classification)-1)  # 类别在self.classification列表中的位置作为标签

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = (torch.Tensor(np.expand_dims(np.array(Image.open(self.x[index][0])), axis=0)).cuda(),
             torch.Tensor(np.expand_dims(np.array(Image.open(self.x[index][1])), axis=0)).cuda(),
             torch.Tensor(np.expand_dims(np.array(Image.open(self.x[index][2])), axis=0)).cuda())
        y = torch.LongTensor(self.y)[index].cuda()
        return x, y

    def getlabel(self):
        return self.classification


class PatchDataTest(Dataset):

    def __init__(self, source=('normal_160', 'sobel_32', 'canny_32')):
        self.x = []
        self.y = []
        self.classification = []
        data_path = ('data/' + source[0] + '/test',
                     'data/' + source[1] + '/test',
                     'data/' + source[2] + '/test')
        for classification in os.listdir(data_path[0]):
            self.classification.append(classification)
            for sample in os.listdir(os.path.join(data_path[0], classification)):
                self.x.append((os.path.join(data_path[0], classification, sample),
                               os.path.join(data_path[1], classification, sample),
                               os.path.join(data_path[2], classification, sample)))
                self.y.append(len(self.classification) - 1)  # 类别在self.classification列表中的位置作为标签

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = (torch.Tensor(np.expand_dims(np.array(Image.open(self.x[index][0])), axis=0)).cuda(),
             torch.Tensor(np.expand_dims(np.array(Image.open(self.x[index][1])), axis=0)).cuda(),
             torch.Tensor(np.expand_dims(np.array(Image.open(self.x[index][2])), axis=0)).cuda())
        y = torch.LongTensor(self.y)[index].cuda()
        return x, y

    def getlabel(self):
        return self.classification
