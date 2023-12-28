# EAST: An Efficient and Accurate Scene Text Detector

PyTorch re-implementation
of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/pdf/1704.03155.pdf) (CVPR 2017).

<div align='center'>
  <img src='assets/east.jpg'>
</div>

## Table of Contents

* [Project Description](#project-description)
* [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)

## Project Description
Training results of EAST model with 600 epochs on **ICDAR2015** dataset.

| Model            | Loss     | Recall    | Precision | F-score   |
|------------------|----------|-----------|-----------|-----------|
| Original         | CE       | 72.75     | 80.46     | 76.41     |
| **Re-Implement** | **Dice** | **81.27** | **80.61** | **80.93** |

## Installation

```
git clone https://github.com/yakhyo/east-pytorch.git
cd east-pytorch 
pip install -r requirements.txt
```

Pre-trained backbone weight can be downloaded from [here](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth) and
put it inside `weights` folder.

### Dataset:

- Download the dataset [here](https://rrc.cvc.uab.es/?ch=4&com=downloads) and place them as shown below:
- Task 4.1: Text Localization (2015 edition)

```
├── east-pytorch
    ├── assets
    ├── evaluate
    ├── east
    |    ├── models
    |    └── utils
    │── weights
    └── data
        ├── ch4_test_gt
        ├── ch4_test_images
        ├── ch4_train_gt
        └── ch4_train_images

```

## Usage

Usage:

```
usage: train.py [-h] [--cfg CFG] [--data-path DATA_PATH] [--pretrained PRETRAINED] [--checkpoint CHECKPOINT] [--save-dir SAVE_DIR]
                [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE] [--num-workers NUM_WORKERS] [--epochs EPOCHS]

```

Train:

```
python train.py
```

Usage of `detect.py`:

```
usage: detect.py [-h] [--cfg CFG] [--weights WEIGHTS] [--input INPUT] [--output OUTPUT]
```

Inference:

```
python detect.py --weights weights/model.pt --input [input_image] --output [output_file_name]
```

<div align="center">
  <img src="assets/output1.png" width="45%">
  <img src="assets/output2.png" width="45%">
</div>

### Evaluation and Inference:

1. The evaluation scripts are from [ICDAR Offline](https://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1) evaluation and
   have been modified to run successfully with Python 3.8.
2. Change the `evaluate/gt.zip` if you test on other datasets.
3. Modify the parameters in `east/tools/eval.py` and run:

## Contributing

If you welcome contributions from other developers, explain how they can get started. Include information on how to
report bugs, request features, or submit code changes. See the [Contributing policy](.github/CONTRIBUTING.md)
and [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## License

The project is licensed under the [MIT license](https://opensource.org/license/mit/).

## Reference
1. Some part of the code refers to https://github.com/SakuraRiven/EAST.
