from .dataUtils import split_ssl_data, sample_labeled_data
from .basicDataset import BasicDataset
import torchvision
import numpy as np
from torchvision import transforms
import json
import os
import random
from .randaugment import RandAugmentMC
import gc
import copy
from PIL import Image

mean, std = {}, {}
mean['cifar10'] = [0.4914, 0.4822, 0.4465]
mean['cifar100'] = [0.5071, 0.4867, 0.4408]
mean['svhn'] = [0.4380, 0.4440, 0.4730]
mean['stl10'] = [0.4408, 0.4278, 0.3867]
mean['imagenet'] = [0.485, 0.456, 0.406]

std['cifar10'] = [0.2471, 0.2435, 0.2616]
std['cifar100'] = [0.2675, 0.2565, 0.2761]
std['svhn'] = [0.1751, 0.1771, 0.1744]
std['stl10'] = [0.2682, 0.2612, 0.2686]
std['imagenet'] = [0.229, 0.224, 0.225]


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImagenetDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, ulb, num_labels=-1):
        super().__init__(root, transform)
        self.ulb = ulb
        self.num_labels = num_labels
        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        if self.ulb:
            self.strong_transform = copy.deepcopy(transform)
            self.strong_transform.transforms.insert(0, RandAugmentMC(2, 10))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample_transformed = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (index, sample_transformed, target) if not self.ulb else (
            index, sample_transformed, self.strong_transform(sample))

    def make_dataset(
            self,
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None,
    ):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)

        lb_idx = {}

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                random.shuffle(fnames)
                if self.num_labels != -1:
                    fnames = fnames[:self.num_labels]
                if self.num_labels != -1:
                    lb_idx[target_class] = fnames
                for fname in fnames:
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
        if self.num_labels != -1:
            with open('./sampled_label_idx.json', 'w') as f:
                json.dump(lb_idx, f)
        del lb_idx
        gc.collect()
        return instances


class ImageNetLoader:
    def __init__(self, root_path, num_labels=-1, num_class=1000):
        self.root_path = os.path.join(root_path, 'imagenet')
        self.num_labels = num_labels // num_class

    def get_transform(self, train, ulb):
        if train:
            transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean["imagenet"], std["imagenet"])])
        else:
            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean["imagenet"], std["imagenet"])])
        return transform

    def get_lb_train_data(self):
        transform = self.get_transform(train=True, ulb=False)
        data = ImagenetDataset(root=os.path.join(self.root_path, "train"), transform=transform, ulb=False,
                               num_labels=self.num_labels)
        return data

    def get_ulb_train_data(self):
        transform = self.get_transform(train=True, ulb=True)
        data = ImagenetDataset(root=os.path.join(self.root_path, "train"), transform=transform, ulb=True)
        return data

    def get_lb_test_data(self):
        transform = self.get_transform(train=False, ulb=False)
        data = ImagenetDataset(root=os.path.join(self.root_path, "val"), transform=transform, ulb=False)
        return data


def get_transform(mean, std, crop_size, train=True):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(crop_size, padding=4, padding_mode='reflect'),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


class SSL_Dataset:

    def __init__(self,
                 args,
                 name='cifar10',
                 train=True,
                 num_classes=10,
                 data_dir='./data'):
        self.args = args
        self.name = name
        self.train = train
        self.num_classes = num_classes
        self.data_dir = data_dir
        crop_size = 96 if self.name.upper() == 'STL10' else 224 if self.name.upper() == 'IMAGENET' else 32
        self.transform = get_transform(mean[name], std[name], crop_size, train)

    def get_data(self, svhn_extra=True):
        dset = getattr(torchvision.datasets, self.name.upper())
        if 'CIFAR' in self.name.upper():
            dset = dset(self.data_dir, train=self.train, download=True)
            data, targets = dset.data, dset.targets
            return data, targets
        elif self.name.upper() == 'SVHN':
            if self.train:
                if svhn_extra:  # train+extra
                    dset_base = dset(self.data_dir, split='train', download=True)
                    data_b, targets_b = dset_base.data.transpose([0, 2, 3, 1]), dset_base.labels
                    dset_extra = dset(self.data_dir, split='extra', download=True)
                    data_e, targets_e = dset_extra.data.transpose([0, 2, 3, 1]), dset_extra.labels
                    data = np.concatenate([data_b, data_e])
                    targets = np.concatenate([targets_b, targets_e])
                    del data_b, data_e
                    del targets_b, targets_e
                else:  # train_only
                    dset = dset(self.data_dir, split='train', download=True)
                    data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
            else:  # test
                dset = dset(self.data_dir, split='test', download=True)
                data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
            return data, targets
        elif self.name.upper() == 'STL10':
            split = 'train' if self.train else 'test'
            dset_lb = dset(self.data_dir, split=split, download=True)
            dset_ulb = dset(self.data_dir, split='unlabeled', download=True)
            data, targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
            ulb_data = dset_ulb.data.transpose([0, 2, 3, 1])
            return data, targets, ulb_data

    def get_dset(self, is_ulb=False,
                 strong_transform=None, onehot=False):
        if self.name.upper() == 'STL10':
            data, targets, _ = self.get_data()
        else:
            data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform

        return BasicDataset(data, targets, num_classes, transform,
                            is_ulb, strong_transform, onehot)

    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
                     strong_transform=None, onehot=False):
        if self.name.upper() == 'STL10':
            lb_data, lb_targets, ulb_data = self.get_data()
            if include_lb_to_ulb:
                ulb_data = np.concatenate([ulb_data, lb_data], axis=0)
            lb_data, lb_targets, _ = sample_labeled_data(self.args, lb_data, lb_targets, num_labels, self.num_classes)
            ulb_targets = None
        else:
            data, targets = self.get_data()
            lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(self.args, data, targets,
                                                                        num_labels, self.num_classes,
                                                                        index, include_lb_to_ulb)


        count = [0 for _ in range(self.num_classes)]
        for c in lb_targets:
            count[c] += 1
        dist = np.array(count, dtype=float)
        dist = dist / dist.sum()
        dist = dist.tolist()
        out = {"distribution": dist}
        output_file = r"./data_statistics/"
        output_path = output_file + str(self.name) + '_' + str(num_labels) + '.json'
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        with open(output_path, 'w') as w:
            json.dump(out, w)
        # print(Counter(ulb_targets.tolist()))
        lb_dset = BasicDataset(lb_data, lb_targets, self.num_classes,
                               self.transform, False, None, onehot)

        ulb_dset = BasicDataset(ulb_data, ulb_targets, self.num_classes,
                                self.transform, True, strong_transform, onehot)

        return lb_dset, ulb_dset
