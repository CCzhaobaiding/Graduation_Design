import argparse
import logging
import math
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import RandomSampler, DistributedSampler, SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.distributed as dist
from dataset.sslDataset import SSL_Dataset, ImageNetLoader
from utils import AverageMeter, accuracy

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
logger = logging.getLogger(__name__)
best_acc = 0


def main():
    parser = argparse.ArgumentParser(description='PyTorch CrossMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'stl10', 'svhn'],
                        help='dataset name')
    parser.add_argument('--num-labels', type=int, default=4000)
    parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
    parser.add_argument('--total-steps', default=2 ** 20, type=int)
    parser.add_argument('--eval-step', default=1024, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float)
    parser.add_argument('--warmup', default=0, type=float)
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True, help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float)
    parser.add_argument('--lambda-dis', default=1, type=float)
    parser.add_argument('--lambda-con', default=1, type=float)
    parser.add_argument('--lambda-com', default=1, type=float)
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    # 锐化温度
    parser.add_argument('--ST', default=0.8, type=float, help='sharpen temperature')
    parser.add_argument('--temperature', default=0.2, type=float, help='softmax temperature')
    parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
    # 图半监督阈值
    parser.add_argument('--contrast-th', default=0.8, type=float, help='pseudo label graph threshold')
    parser.add_argument('--out', default='result', help='directory to output the result')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=5, type=int, help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    args = parser.parse_args()

    if args.dataset == 'imagenet':
        args.num_classes = 1000
        args.model_depth = 0
        args.model_width = 0
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.model_depth = 28
        args.model_width = 8
    else:
        args.num_classes = 10
        args.model_depth = 28
        args.model_width = 2

    # choose model
    def create_model(args):
        if args.dataset == 'imagenet':
            import models.resnet50 as models
            model = models.build_ResNet50(num_classes=args.num_classes)
        elif args.dataset == 'stl10':
            import models.wideresnet_var as models
            model = models.build_WideResNetVar(depth=args.model_depth,
                                               widen_factor=args.model_width,
                                               dropout=0,
                                               num_classes=args.num_classes)
        else:
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}")

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset != "imagenet":
        train_dset = SSL_Dataset(args, name=args.dataset, train=True,
                                 num_classes=args.num_classes, data_dir=args.data_dir)
        labeled_dataset, unlabeled_dataset = train_dset.get_ssl_dset(args.num_labels)
        _test_dset = SSL_Dataset(args, name=args.dataset, train=False,
                                 num_classes=args.num_classes, data_dir=args.data_dir)
        test_dataset = _test_dset.get_dset()
    else:
        image_loader = ImageNetLoader(root_path=args.data_dir, num_labels=args.num_labels,
                                      num_class=args.num_classes)
        labeled_dataset = image_loader.get_lb_train_data()
        unlabeled_dataset = image_loader.get_ulb_train_data()
        test_dataset = image_loader.get_lb_test_data()

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model1 = create_model(args)
    model2 = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model1.to(args.device)
    model2.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters1 = [
        {'params': [p for n, p in model1.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model1.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer1 = optim.SGD(grouped_parameters1, lr=args.lr,
                           momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler1 = get_cosine_schedule_with_warmup(
        optimizer1, args.warmup, args.total_steps)

    grouped_parameters2 = [
        {'params': [p for n, p in model2.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model2.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer2 = optim.SGD(grouped_parameters2, lr=args.lr,
                           momentum=0.9, nesterov=args.nesterov)

    scheduler2 = get_cosine_schedule_with_warmup(
        optimizer2, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model1 = ModelEMA(args, model1, args.ema_decay)
        ema_model2 = ModelEMA(args, model2, args.ema_decay)

    args.start_epoch = 0
    global best_acc
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model1.load_state_dict(checkpoint['state_dict1'])
        model2.load_state_dict(checkpoint['state_dict2'])
        if args.use_ema:
            ema_model1.ema.load_state_dict(checkpoint['ema_state_dict1'])
            ema_model2.ema.load_state_dict(checkpoint['ema_state_dict2'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        scheduler1.load_state_dict(checkpoint['scheduler1'])
        scheduler2.load_state_dict(checkpoint['scheduler2'])

    if args.local_rank != -1:
        model1 = torch.nn.parallel.DistributedDataParallel(
            model1, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        model2 = torch.nn.parallel.DistributedDataParallel(
            model2, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labels}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model1.zero_grad()
    model2.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model1, optimizer1, ema_model1, scheduler1, model2, optimizer2, ema_model2, scheduler2)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model1, optimizer1, ema_model1, scheduler1, model2, optimizer2, ema_model2, scheduler2):
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model1.train()
    model2.train()
    for epoch in range(args.start_epoch, args.epochs):
        logger.info("Train Epoch: {:}/1024".format(epoch))

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_labeled = AverageMeter()
        losses_unlabeled = AverageMeter()
        losses_cross_dis = AverageMeter()
        losses_cross_con = AverageMeter()
        losses_graph_com = AverageMeter()
        mask_probs1 = AverageMeter()
        mask_probs2 = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])

        # for distribution alignment
        pseudo_label_list1 = []
        pseudo_label_list2 = []

        for batch_idx in range(args.eval_step):
            try:
                _, inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                _, inputs_x, targets_x = labeled_iter.next()
            try:
                _, inputs_u_w, inputs_u_s1, inputs_u_s2 = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                _, inputs_u_w, inputs_u_s1, inputs_u_s2 = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs1 = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s1)), 2 * args.mu + 1).to(args.device)
            inputs2 = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s2)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits1, features1 = model1(inputs1)
            logits1 = de_interleave(logits1, 2 * args.mu + 1)
            logits_x1 = logits1[:batch_size]
            logits_u_w1, logits_u_s1 = logits1[batch_size:].chunk(2)
            del logits1
            logits2, features2 = model2(inputs2)
            logits2 = de_interleave(logits2, 2 * args.mu + 1)
            logits_x2 = logits2[:batch_size]
            logits_u_w2, logits_u_s2 = logits2[batch_size:].chunk(2)
            del logits2

            pseudo_label1 = torch.softmax(logits_u_w1.detach() / args.T, dim=-1)
            pseudo_label2 = torch.softmax(logits_u_w2.detach() / args.T, dim=-1)

            # 嵌入的切分
            features1 = de_interleave(features1, 2 * args.mu + 1)
            _, feats_u_s1 = features1[batch_size:].chunk(2)
            feats_u_s1 = feats_u_s1.detach()
            features2 = de_interleave(features2, 2 * args.mu + 1)
            _, feats_u_s2 = features2[batch_size:].chunk(2)
            feats_u_s2 = feats_u_s2.detach()

            # DA
            pseudo_label_list1.append(pseudo_label1.mean(0))
            pseudo_label_list2.append(pseudo_label2.mean(0))
            if len(pseudo_label_list1) > 128:
                pseudo_label_list1.pop(0)
                pseudo_label_list2.pop(0)
            pseudo_label_avg1 = torch.stack(pseudo_label_list1, dim=0).mean(0)
            pseudo_label1 = pseudo_label1 / pseudo_label_avg1
            pseudo_label1 = pseudo_label1 / pseudo_label1.sum(dim=1, keepdim=True)
            pseudo_label_avg2 = torch.stack(pseudo_label_list2, dim=0).mean(0)
            pseudo_label2 = pseudo_label2 / pseudo_label_avg2
            pseudo_label2 = pseudo_label2 / pseudo_label2.sum(dim=1, keepdim=True)

            # sharpen
            pseudo_label1 = pseudo_label1 ** (1 / args.ST)
            pseudo_label1 = pseudo_label1 / pseudo_label1.sum(dim=1, keepdim=True)
            pseudo_label2 = pseudo_label2 ** (2 / args.ST)
            pseudo_label2 = pseudo_label2 / pseudo_label2.sum(dim=1, keepdim=True)

            # max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            max_probs1, _ = torch.max(pseudo_label1, dim=-1)
            mask1 = max_probs1.ge(args.threshold).float()
            max_probs2, _ = torch.max(pseudo_label2, dim=-1)
            mask2 = max_probs2.ge(args.threshold).float()

            # supervised classification loss
            loss_labeled1 = F.cross_entropy(logits_x1, targets_x, reduction='mean')
            loss_labeled2 = F.cross_entropy(logits_x2, targets_x, reduction='mean')
            loss_labeled = (loss_labeled1 + loss_labeled2) / 2

            # unsupervised cross-entropy loss
            loss_unlabeled1 = (torch.sum(-F.log_softmax(logits_u_s1, dim=1) * pseudo_label1, dim=1) * mask1).mean()
            loss_unlabeled2 = (torch.sum(-F.log_softmax(logits_u_s2, dim=1) * pseudo_label2, dim=1) * mask2).mean()
            loss_unlabeled = (loss_unlabeled1 + loss_unlabeled2) / 2
            loss_unlabeled = args.lambda_u * loss_unlabeled

            # discrepancy loss
            cos_dis = nn.CosineSimilarity(dim=1, eps=1e-6)

            # cross supervised difference loss
            loss_cross_labeled1 = 1 + cos_dis(logits_x1.detach(), logits_x2).mean()
            loss_cross_labeled2 = 1 + cos_dis(logits_x2.detach(), logits_x1).mean()
            loss_cross_labeled = (loss_cross_labeled1 + loss_cross_labeled2) / 2

            # cross unsupervised difference loss
            loss_cross_weak_unlabeled1 = 1 + cos_dis(logits_u_w1.detach(), logits_u_w2).mean()
            loss_cross_weak_unlabeled2 = 1 + cos_dis(logits_u_w2.detach(), logits_u_w1).mean()
            loss_cross_weak_unlabeled = (loss_cross_weak_unlabeled1 + loss_cross_weak_unlabeled2) / 2
            loss_cross_strong_unlabeled1 = 1 + cos_dis(logits_u_s1.detach(), logits_u_s2).mean()
            loss_cross_strong_unlabeled2 = 1 + cos_dis(logits_u_s2.detach(), logits_u_s1).mean()
            loss_cross_strong_unlabeled = (loss_cross_strong_unlabeled1 + loss_cross_strong_unlabeled2) / 2
            loss_cross_unlabeled = (loss_cross_weak_unlabeled + loss_cross_strong_unlabeled) / 2

            # cross supervised difference loss
            loss_cross_dis = (loss_cross_labeled + loss_cross_unlabeled) / 2
            loss_cross_dis = args.lambda_dis * loss_cross_dis

            # cross enforce consistence loss
            loss_cross_con1 = (torch.sum(-F.log_softmax(logits_u_s1, dim=1) * pseudo_label2, dim=1) * mask1).mean()
            loss_cross_con2 = (torch.sum(-F.log_softmax(logits_u_s2, dim=1) * pseudo_label1, dim=1) * mask2).mean()
            loss_cross_con = (loss_cross_con1 + loss_cross_con2) / 2
            loss_cross_con = args.lambda_con * loss_cross_con

            # collaborative graph comparison Loss 协同图对比损失
            # embedding similarity graph
            sim = torch.exp(torch.mm(feats_u_s1, feats_u_s2.t()) / args.temperature)
            sim_probs = sim / sim.sum(1, keepdim=True)
            # pseudo-label graph
            Q = torch.mm(pseudo_label1, pseudo_label2.t())
            Q.fill_diagonal_(1)
            pos_mask = (Q >= args.contrast_th).float()
            Q = Q * pos_mask
            Q = Q / Q.sum(1, keepdim=True)
            loss_graph_com = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
            loss_graph_com = loss_graph_com.mean()
            loss_graph_com = args.lambda_com * loss_graph_com

            loss = loss_labeled + loss_unlabeled + loss_cross_dis + loss_cross_con + loss_graph_com

            loss.backward()
            losses.update(loss.item())
            losses_labeled.update(loss_labeled.item())
            losses_unlabeled.update(loss_unlabeled.item())
            losses_cross_dis.update(loss_cross_dis.item())
            losses_cross_con.update(loss_cross_con.item())
            losses_graph_com.update(loss_graph_com.item())

            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()

            if args.use_ema:
                ema_model1.update(model1)
                ema_model2.update(model2)
            model1.zero_grad()
            model2.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs1.update(mask1.mean().item())
            mask_probs2.update(mask1.mean().item())

            # if not args.no_progress:
            #     p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR1: {lr1:.4f}. LR2: {lr2:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_labeled: {loss_labeled:.4f}. Loss_unlabeled: {loss_unlabeled:.4f}. Loss_cross_dif: {loss_cross_dif:.4f}. Loss_cross_con: {loss_cross_con:.4f}. Loss_graph_com: {loss_graph_com:.4f}. Mask1: {mask1:.2f}. Mask2: {mask2:.2f}.".format(
            #         epoch=epoch + 1,
            #         epochs=args.epochs,
            #         batch=batch_idx + 1,
            #         iter=args.eval_step,
            #         lr1=scheduler1.get_last_lr()[0],
            #         lr2=scheduler2.get_last_lr()[0],
            #         data=data_time.avg,
            #         bt=batch_time.avg,
            #         loss=losses.avg,
            #         loss_labeled=losses_labeled.avg,
            #         loss_unlabeled=losses_unlabeled.avg,
            #         loss_cross_dif=losses_cross_dif.avg,
            #         loss_cross_con=losses_cross_con.avg,
            #         loss_graph_com=losses_graph_com.avg,
            #         mask1=mask_probs1.avg,
            #         mask2=mask_probs2.avg))
            #     p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model1.ema
        else:
            test_model = model1

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_labeled', losses_labeled.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_unlabeled', losses_unlabeled.avg, epoch)
            args.writer.add_scalar('train/4.train_loss_cross_dif', losses_cross_dis.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_cross_con', losses_cross_con.avg, epoch)
            args.writer.add_scalar('train/6.train_loss_graph_com', losses_graph_com.avg, epoch)
            args.writer.add_scalar('train/7.mask1', mask_probs1.avg, epoch)
            args.writer.add_scalar('train/8.mask2', mask_probs2.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save1 = model1.module if hasattr(model1, "module") else model1
            model_to_save2 = model2.module if hasattr(model2, "module") else model2
            if args.use_ema:
                ema_to_save1 = ema_model1.ema.module if hasattr(
                    ema_model1.ema, "module") else ema_model1.ema
                ema_to_save2 = ema_model2.ema.module if hasattr(
                    ema_model2.ema, "module") else ema_model2.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict1': model_to_save1.state_dict(),
                'state_dict2': model_to_save2.state_dict(),
                'ema_state_dict1': ema_to_save1.state_dict() if args.use_ema else None,
                'ema_state_dict2': ema_to_save2.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'scheduler1': scheduler1.state_dict(),
                'scheduler2': scheduler2.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for (batch_idx, inputs, targets) in test_loader:
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs, _ = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
        #     if not args.no_progress:
        #         test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
        #             batch=batch_idx + 1,
        #             iter=len(test_loader),
        #             data=data_time.avg,
        #             bt=batch_time.avg,
        #             loss=losses.avg,
        #             top1=top1.avg,
        #             top5=top5.avg,
        #         ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
