import os
import time
import shutil
import subprocess

import torch

from east.models import EAST
from detect import detect_dataset


def eval_model(model_path, test_img_path, submit_path, save_flag=True):
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.mkdir(submit_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(False).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    start_time = time.time()
    detect_dataset(model, device, test_img_path, submit_path)
    os.chdir(submit_path)
    res = subprocess.getoutput('zip -q submit.zip *.txt')
    res = subprocess.getoutput('mv submit.zip ../')
    os.chdir('../')
    res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
    print(res)
    os.remove('./submit.zip')
    print('eval time is {}'.format(time.time() - start_time))

    if not save_flag:
        shutil.rmtree(submit_path)


if __name__ == '__main__':
    weights = './weights/model_epoch_600.pth'
    test_img_path = os.path.abspath('data/ch4_test_images')
    submit_path = './submit'
    print("Evaluation started...")
    start = time.time()
    eval_model(weights, test_img_path, submit_path)
    print(f"Evaluation finished in {(time.time() - start)}s")
