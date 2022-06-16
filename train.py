import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time

import argparse
import temp


def train(args):
    file_num = len(os.listdir(args.train_images))
    trainset = custom_dataset(args.train_images, args.train_labels)
    train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   drop_last=True)

    criterion = Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EAST()
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 2], gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        scheduler.step()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(
                device)
            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
                epoch + 1, args.epochs, i + 1, int(file_num / args.batch_size), time.time() - start_time, loss.item()))

        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / int(file_num / args.batch_size),
                                                                  time.time() - epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('=' * 50)
        if (epoch + 1) % args.interval == 0:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(args.weights, 'model_epoch_{}.pth'.format(epoch + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EAST: An Efficient and Accurate Scene Text Detector")

    parser.add_argument("--train-images", default="data/ch4_training_images", help="Path to training images")
    parser.add_argument("--train-labels", default="data/ch4_training_gt", help="Path to training labels")
    parser.add_argument("--weights", default="./weights", help="Path to weights folder")
    parser.add_argument("--batch-size", default=20, type=int, help="Batch size")
    parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--num-workers", default=8, type=int, help="Number of workers")
    parser.add_argument("--epochs", default=600, type=int, help="Number of epochs")
    parser.add_argument("--interval", default=50, type=int, help="Interval to save weights")

    config = parser.parse_args()

    train(config)
