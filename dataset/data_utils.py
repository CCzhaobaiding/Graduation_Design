import logging
import math
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from .randaugment import RandAugmentMC

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)

        return image, target


def x_u_split(args, labels):
    label_per_class = args.num_labels // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labels

    if args.expand_labels or args.num_labels < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labels)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformSSL(object):
    def __init__(self, mean, std, crop):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=crop,
                                  padding=int(4),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=crop,
                                  padding=int(4),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong1 = self.strong(x)
        strong2 = self.strong(x)
        return weak, strong1, strong2