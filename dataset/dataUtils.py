import math
import torch
from torch.utils.data import sampler
import numpy as np


def split_ssl_data(args, data, target, num_labels, num_classes, index=None):
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx, = sample_labeled_data(args, data, target, num_labels, num_classes, index)
    ulb_idx = np.array(range(len(target)))
    # 扩增到64*7*1024后全部一起RandomSampler还是保持原大小散列后iterator重复读取，两种结果实验对比
    return lb_data, lbs, data[ulb_idx], target[ulb_idx]


def sample_labeled_data(args, data, target,
                        num_labels, num_classes,
                        index=None, name=None):
    assert num_labels % num_classes == 0
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    samples_per_class = int(num_labels / num_classes)

    lb_data = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

    lb_idx = np.array(lb_idx)
    if args.expand_labels or args.num_labels < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labels)
        lb_idx = np.hstack([lb_idx for _ in range(num_expand_x)])
    # 实验对比分布式训练把数据扩增到*world_size好不好
    np.random.shuffle(lb_idx)

    lb_data.extend(data[lb_idx])
    lbs.extend(target[lb_idx])

    return np.array(lb_data), np.array(lbs), lb_idx


def get_sampler_by_name(name):
    sampler_name_list = sorted(name for name in torch.utils.data.sampler.__dict__
                               if not name.startswith('_') and callable(sampler.__dict__[name]))
    try:
        if name == 'DistributedSampler':
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)
    except Exception as e:
        print(repr(e))
        print('[!] select sampler in:\t', sampler_name_list)


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot