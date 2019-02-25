from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from captchaRecognition.config import DefaultConfig


class dataset(Dataset):

    def __init__(self, root_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.label = np.loadtxt(label_file)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, '{0:0>4}'.format(idx) + '.jpg')
        image = Image.open(img_name)
        labels = self.label[idx, :]
        if self.transform:
            image = self.transform(image)
        return image, labels

    def __len__(self):
        return (self.label.shape[0])


data = dataset(DefaultConfig.file_path + 'src', DefaultConfig.file_path + 'label.txt', transform=transforms.ToTensor())

data_loader = DataLoader(data,
                         batch_size=DefaultConfig.BATCH_SIZE,
                         shuffle=True,
                         num_workers=4,
                         drop_last=True)

dataset_size = len(data)
