import os
import time
import shutil
import argparse
import subprocess

import torch

from nets.nn import EAST
from detect import detect_dataset


def eval_model(opt):
    print('Evaluating...')
    if os.path.exists(opt.submit):
        shutil.rmtree(opt.submit)
    os.mkdir(opt.submit)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(False).to(device)
    model.load_state_dict(torch.load(opt.saved_model))
    model.eval()

    start_time = time.time()
    detect_dataset(model, device, opt.test_images, opt.submit)
    os.chdir(opt.submit)
    res = subprocess.getoutput('zip -q submit.zip *.txt')
    res = subprocess.getoutput('mv submit.zip ../')
    os.chdir('../')
    res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
    print(res)
    os.remove('./submit.zip')
    print('Evaluation finished in {}'.format(time.time() - start_time))

    if not opt.save_flag:
        shutil.rmtree(opt.submit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EAST: An Efficient and Accurate Scene Text Detector')
    parser.add_argument('--saved_model', type=str, default='./weights/east.pth', help='path to saved model')
    parser.add_argument('--test_images', type=str, default='../data/ICDAR_2015/test_img', help='path to test images')
    parser.add_argument('--submit', type=str, default='./submit', help='path to save results')
    parser.add_argument('--save_flag', type=bool, default=True, help='path to save results')

    opt = parser.parse_args()

    eval_model(opt)
