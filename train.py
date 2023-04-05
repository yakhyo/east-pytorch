import argparse
import logging
import os
import warnings
from copy import deepcopy

import torch
from torch.optim import lr_scheduler
from tqdm import tqdm

from east.models import EAST
from east.utils.dataset import create_dataloader
from east.utils.loss import Loss
from east.utils.misc import strip_optimizer

warnings.simplefilter("ignore")


def train(opt, model, device):
    start_epoch = 0
    os.makedirs(opt.save_dir, exist_ok=True)

    # Check checkpoints
    pretrained = opt.checkpoint.endswith(".ckpt")
    if pretrained:
        ckpt = torch.load(opt.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"].float().state_dict())
        logging.info(f"Model ckpt loaded from {opt.checkpoint}")

    logging.info("Creating Dataloader")
    train_loader = create_dataloader(opt.data_path, batch_size=opt.batch_size, num_workers=opt.num_workers)

    criterion = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[opt.epochs // 2], gamma=0.1)

    # Resume
    if pretrained:
        if ckpt["optimizer"] is not None:
            start_epoch = ckpt["epoch"] + 1
            optimizer.load_state_dict(ckpt["optimizer"])
            logging.info(f"Optimizer loaded from {opt.checkpoint}")
            if start_epoch < opt.epochs:
                logging.info(
                    f"{opt.checkpoint} has been trained for {start_epoch} epochs. Fine-tuning for {opt.epochs} epochs"
                )
        del ckpt

    for epoch in range(start_epoch, opt.epochs):
        model.train()
        logging.info(("\n" + "%12s" * 4) % ("Epoch", "GPU Mem", "Geo Loss", "Dice Loss"))
        progress_bar = tqdm(train_loader, total=len(train_loader))
        for image, gt_score, gt_geo, ignored_map in progress_bar:
            image, gt_score, gt_geo, ignored_map = (
                image.to(device),
                gt_score.to(device),
                gt_geo.to(device),
                ignored_map.to(device),
            )
            pred_score, pred_geo = model(image)
            loss_dict = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
            loss = loss_dict["geo_loss"] + loss_dict["cls_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            progress_bar.set_description(
                ("%12s" * 2 + "%12.4g" * 2) % (
                    f"{epoch + 1}/{opt.epochs}", mem, loss_dict["geo_loss"], loss_dict["cls_loss"])
            )

        scheduler.step()
        ckpt = {
            "epoch": epoch,
            "model": deepcopy(model).half(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, f"{opt.save_dir}/model.ckpt")

    strip_optimizer(f"{opt.save_dir}/model.ckpt")


def main(opt):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("train: " + ", ".join(f"{k}={v}" for k, v in vars(opt).items()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model = EAST(cfg=opt.cfg, weights=opt.pretrained)
    model.to(device)

    train(opt, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EAST: An Efficient and Accurate Scene Text Detector")
    parser.add_argument("--cfg", default="D", type=str, help="VGG backbone config | [A, B, D, E]")
    parser.add_argument("--data-path", default="data/ch4_train_images", help="Path to training images")
    parser.add_argument("--pretrained", default="./weights/vgg16_bn-6c64b313.pth", type=str,
                        help="Pretrained backbone path | None")
    parser.add_argument("--checkpoint", default="./weights/model.ckpt", type=str,
                        help="Continue the training from checkpoint")
    parser.add_argument("--save-dir", default="./weights", help="Path to saving weights")
    parser.add_argument("--batch-size", default=20, type=int, help="Batch size")
    parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--num-workers", default=8, type=int, help="Number of workers")
    parser.add_argument("--epochs", default=600, type=int, help="Number of epochs")
    args = parser.parse_args()

    main(args)
