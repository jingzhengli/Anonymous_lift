## LiFT: Transfer Learning in Vision-Language Models for Downstream Adaptation and Generalization

### Introduction
This is a PyTorch implementation of ["LiFT: Transfer Learning in Vision-Language Models for Downstream Adaptation and Generalization"]. 

### Requirements
* pytorch 1.10.0
* timm 0.4.12
* tensorboardX
* ftfy
* dassl

### Data preparation
Please follow the instructions at [DATASETS.md](DATASETS.md) to prepare all datasets.

### Training:

- LiFT over 11 datasets:
    ```bash
    sh train.sh
    ```

- LiFT-Adapter over 11 datasets:
    ```bash
    sh train_liftadapter.sh
    ```

- LiFT-NCD over 11 datasets:

    ```bash
    sh train_liftncd.sh
    ```
