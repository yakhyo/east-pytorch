import os
import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import lr_scheduler

from nets.nn import EAST
from util.loss import Loss
from util.dataset import EASTDataset


def train(opt):
    file_num = len(os.listdir(opt.train_images))
    dataset = EASTDataset(opt.train_images, opt.train_labels)
    train_loader = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                                   drop_last=True)

    criterion = Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EAST()
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[opt.epoch_iter // 2], gamma=0.1)

    for epoch in range(opt.epoch_iter):
        model.train()
        epoch_loss = 0
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (img, gt_score, gt_geo, ignored_map) in progress_bar:
            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(
                device)
            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, opt.epoch_iter), epoch_loss / (i + 1), mem)
            progress_bar.set_description(s)

        scheduler.step()

        if (epoch + 1) % opt.interval == 0:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(opt.model_save, 'model_epoch_{}.pth'.format(epoch + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EAST: An Efficient and Accurate Scene Text Detector')
    parser.add_argument('--train_images', type=str, default='../data/ICDAR_2015/train_img', help='path to train images')
    parser.add_argument('--train_labels', type=str, default='../data/ICDAR_2015/train_gt', help='path to train labels')
    parser.add_argument('--model_save', type=str, default='./weights', help='path to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers in dataloader')
    parser.add_argument('--epoch_iter', type=int, default=600, help='number of iterations')
    parser.add_argument('--interval', type=int, default=50, help='saving interval of checkpoints')

    opt = parser.parse_args()
    train(opt)
