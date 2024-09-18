from torchvision import datasets, transforms
from torch import tensor, long
import os

def DomainNet(data_path):
    channel = 3
    im_size = (224, 224)
    num_classes = 345
    mean = [0.48233780, 0.44213275, 0.41226821]
    std = [0.25210841, 0.22401817, 0.21974742]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    dst_train = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transform)
    dst_test = datasets.ImageFolder(root=os.path.join(data_path, 'test'), transform=transform)
    
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
