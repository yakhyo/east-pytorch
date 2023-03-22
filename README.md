## EAST: An Efficient and Accurate Scene Text Detector

PyTorch re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/pdf/1704.03155.pdf).

<div align='center'>
  <img src='assets/east.jpg'>
</div>



## Description:
*after 600 epochs:

| Model | Loss | Recall   | Precision | F-score   |
|-------|------|----------|-----------|-----------|
| Original | CE | 72.75    | 80.46     | 76.41     |
| **Re-Implement** | **Dice** | **76.6** | **80.92** | **78.70** |

Run:
```
 git clone https://github.com/yakhyo/EAST-pt.git
 cd EAST-pt
 python train.py
```

## Content:

- Data
- Training
- Evaluation and Inference

## Data:

- Download the dataset [here](https://rrc.cvc.uab.es/?ch=4&com=downloads) and place them as shown below:
- Task 4.1: Text Localization (2015 edition)
```
.
├── EAST-pt
    ├── assets
    ├── evaluate
    ├── east
    |    ├── loss
    |    ├── models
    |    ├── tools
    |    └── utils
    │── weights
    └── data
        ├── ch4_test_gt
        ├── ch4_test_images
        ├── ch4_train_gt
        └── ch4_train_images

```

## Training:

Modify the parameters in `east/tools/train.py` and run inside `tools` folder:

```
python train.py
```

## Evaluation and Inference:
1. The evaluation scripts are from [ICDAR Offline](https://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1) evaluation and have been modified to run successfully with Python 3.8.
2. Change the `evaluate/gt.zip` if you test on other datasets.
3. Modify the parameters in `east/tools/eval.py` and run:

- **Detect:**

Modify the parameters in `east/tools/detect.py` and run:
  ```
  python detect.py
  ```

<div align="center">
  <img src="assets/res.bmp" width="45%">
  <img src="assets/res.png" width="45%">
</div>

## Reference
1. The code refers to https://github.com/SakuraRiven/EAST with some modifications.
