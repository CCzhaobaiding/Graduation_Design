from torchvision import transforms
from torch.utils.data import Dataset
from .randaugment import RandAugmentMC
from .dataUtils import get_onehot
from PIL import Image
import numpy as np
import copy


class BasicDataset(Dataset):
    def __init__(self,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):

        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugmentMC(2, 10))
        else:
            self.strong_transform = strong_transform

    def __getitem__(self, idx):

        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.is_ulb:
                return idx, img_w, target
            else:
                return idx, img_w, self.strong_transform(img), self.strong_transform(img)

    def __len__(self):
        return len(self.data)
