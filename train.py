import argparse
import os
import time

import torch
from east.models import EAST
from east.utils.dataset import Dataset

from east.utils.loss import Loss
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data


def train(args):
    file_num = len(os.listdir(args.train_images))
    os.makedirs(args.weights, exist_ok=True)
    print("Initializing Dataset...")
    start = time.time()
    dataset = Dataset(args.train_images, args.train_labels)
    print(f"Initialized in {(time.time()-start)*1000}ms")

    print("Creating Dataloader...")
    start = time.time()
    train_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True
    )
    print(f"Created in {(time.time() - start)*1000}ms")
    criterion = Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating Model...")
    model = EAST(cfg="D", weights="./weights/vgg16_bn.pth")
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    print("Model created! Training getting started")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 2], gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_time = time.time()
        for idx, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = (
                img.to(device),
                gt_score.to(device),
                gt_geo.to(device),
                ignored_map.to(device),
            )
            pred_score, pred_geo = model(img)
            loss_dict = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
            loss = loss_dict["geo_loss"] + loss_dict["classify_loss"]

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                "Epoch: [{}/{} ({}/{})]\t"
                "Time: {:>4.4f}ms\t"
                "Geo Loss: {:>4.4f}\t"
                "Class Loss: {:>4.4f}".format(
                    epoch,
                    args.epochs,
                    idx,
                    file_num // args.batch_size,
                    (time.time() - start_time) * 1000,
                    loss_dict["geo_loss"].item(),
                    loss_dict["classify_loss"],
                )
            )

        scheduler.step()
        print(
            "Epoch Loss is {:.8f}, epoch_time is {:.8f}".format(
                epoch_loss / (file_num // args.batch_size), time.time() - epoch_time
            )
        )
        print(time.asctime(time.localtime(time.time())))
        print("=" * 50)
        if (epoch + 1) % args.interval == 0:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(args.weights, "model_epoch_{}.pth".format(epoch + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EAST: An Efficient and Accurate Scene Text Detector")

    parser.add_argument("--train-images", default="data/ch4_train_images", help="Path to training images")
    parser.add_argument("--train-labels", default="data/ch4_train_gt", help="Path to training labels")
    parser.add_argument("--weights", default="./weights", help="Path to weights folder")
    parser.add_argument("--batch-size", default=20, type=int, help="Batch size")
    parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--num-workers", default=8, type=int, help="Number of workers")
    parser.add_argument("--epochs", default=600, type=int, help="Number of epochs")
    parser.add_argument("--interval", default=10, type=int, help="Interval to save weights")

    config = parser.parse_args()

    train(config)
