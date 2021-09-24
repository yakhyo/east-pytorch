## EAST: An Efficient and Accurate Scene Text Detector

PyTorch re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/pdf/1704.03155.pdf).


<div align='center'>
  <img src='assets/east.jpg'>
</div>



## Description:

| Model | Loss | Recall | Precision | F-score |
|-------|------|--------|-----------|---------|
| Original | CE | 72.75 | 80.46 | 76.41 |
| Re-Implement | Dice | 81.65 | 80.98 | 81.31 |


## Content:

- Data
- Training
- Evaluation and Inference

## Data:

- Download the training files from [here](https://rrc.cvc.uab.es/?ch=4&com=downloads) and place them as shown below:
- Task 4.1: Text Localization (2015 edition)
```
.
├── EAST
│   ├── evaluate
│   ├── nets
│   ├── utils
│   └── weights
└── ICDAR_2015
    ├── test_gt
    ├── test_img
    ├── train_gt
    └── train_img

```

## Training:

Modify the parameters in `train.py` and run:

```
python train.py
```

## Evaluation and Inference:
1. The evaluation scripts are from [ICDAR Offline](https://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1) evaluation and have been modified to run successfully with Python 3.8.
2. Change the `evaluate/gt.zip` if you test on other datasets.
3. Modify the parameters in `eval.py` and run:

- **Detect:**

    Modify the parameters in `detect.py` and run:
  ```
  python detect.py
  ```

<div align='center'>
  <img src='assets/res.bmp'>
</div>

## Reference
1. The code refers to https://github.com/SakuraRiven/EAST with some modifications.