import os
import random

from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt


def cutmix(img1, img2, lab1, lab2):
    a = random.uniform(0, 1)
    if a >= 0.15:
        lab = lab1
    else:
        scale = img1.shape[0]
        x, y = random.randint(3, scale - 3), random.randint(3, scale - 3)
        w, h = random.randint(3, min(x, scale - x)), random.randint(3, min(y, scale - y))
        p1, p2 = [x - w, y - h], [x + w, y + h]
        img1[p1[0]:p2[0], p1[1]:p2[1]] = img2[p1[0]:p2[0], p1[1]:p2[1]]
        alpha = (p2[0]-p1[0])*(p2[1]-p1[1])/(scale**2)
        lab = lab1*(1-alpha) + lab2*alpha
    return img1, lab


def cutout(img):
    a = random.uniform(0, 1)
    if a >= 0.15:
        pass
    else:
        scale = img.shape[0]
        x, y = random.randint(3, scale - 3), random.randint(3, scale - 3)
        w, h = random.randint(3, min(x, scale - x)), random.randint(3, min(y, scale - y))
        p1, p2 = [x - w, y - h], [x + w, y + h]
        img[p1[0]:p2[0], p1[1]:p2[1]] = 0
    return img


def mixup(img1, img2, lab1, lab2):
    a = random.uniform(0, 1)
    if a >= 0.15:
        return img1, lab1
    else:
        alpha = random.uniform(0, 1)
        img = (img1*alpha + img2*(1-alpha)).astype(np.uint8)
        lab = lab1*alpha + lab2*(1-alpha)
    return img, lab


class mydataset(Dataset):
    def __init__(self, data, train=True, transforms=None, method='None'):
        self.data = data  #传入的train或者test数据集
        self.train = train
        self.transforms = transforms
        self.images = self.data['data']
        self.labels = self.data['fine_labels']
        self.method = method

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        total_index = len(self.images)
        if self.method == 'None':
            img = np.reshape(self.images[index], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            img = Image.fromarray(np.uint8(img))
            lab = np.eye(100)[self.labels[index]]
            # lab = self.labels[index]
        elif self.method == 'cutmix':
            img1 = np.reshape(self.images[index], (3, 32, 32))
            img1 = img1.transpose(1, 2, 0)
            lab1 = np.eye(100)[self.labels[index]]
            new_index = random.sample(set(range(0, total_index))-set(range(index, index+1)), 1)[0]
            img2 = np.reshape(self.images[new_index], (3, 32, 32))
            img2 = img2.transpose(1, 2, 0)
            lab2 = np.eye(100)[self.labels[new_index]]
            img, lab = cutmix(img1, img2, lab1, lab2)
            img = Image.fromarray(np.uint8(img))
        elif self.method == 'cutout':
            img = np.reshape(self.images[index], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            lab = np.eye(100)[self.labels[index]]
            img = cutout(img)
            img = Image.fromarray(np.uint8(img))
        elif self.method == 'mixup':
            img1 = np.reshape(self.images[index], (3, 32, 32))
            img1 = img1.transpose(1, 2, 0)
            lab1 = np.eye(100)[self.labels[index]]
            new_index = random.sample(set(range(0, total_index)) - set(range(index, index + 1)), 1)[0]
            img2 = np.reshape(self.images[new_index], (3, 32, 32))
            img2 = img2.transpose(1, 2, 0)
            lab2 = np.eye(100)[self.labels[new_index]]
            img, lab = mixup(img1, img2, lab1, lab2)
            img = Image.fromarray(np.uint8(img))

        # if self.train:
        #     train_transform = T.Compose([
        #         T.ColorJitter(brightness=.5, hue=.3),
        #         # T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
        #         T.RandomRotation(degrees=(0, 180)),
        #         T.RandomPosterize(bits=2),
        #         T.ToTensor(),
        #     ])
        # else:
        train_transform = T.ToTensor()
        img = train_transform(img)

        return img, lab


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
#     print(img)
        dict = pickle.load(fo, encoding='latin1')
    return dict


# cifar100_train = unpickle('../data/cifar-100-python/train')
# cifar100_test = unpickle('../data/cifar-100-python/test')
# print(cifar100_test.keys())
# train_data = mydataset(cifar100_train, method='mixup')
# for data in train_data:
#     img, lab = data
#     plt.imshow(img)
#     plt.show()
#     # print(lab)