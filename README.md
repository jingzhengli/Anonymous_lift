## LiFT: Transfer Learning in Vision-Language Models for Downstream Adaptation and Generalization

### Introduction
This is a PyTorch implementation of ["LiFT: Transfer Learning in Vision-Language Models for Downstream Adaptation and Generalization"]. 

### Requirements
* pytorch 1.12.1
* timm 0.4.12
* tensorboardX
* ftfy
* dassl

### Data preparation
Data preparation is borrowed from a baseline CoOp. Please follow the instructions at [DATASETS.md](DATASETS.md) to prepare all datasets.

### Training:

- Run LiFT on 11 datasets:
    ```bash
    sh train.sh
    ```

- Run LiFT-Adapter on 11 datasets:
    ```bash
    sh train_liftadapter.sh
    ```

- Run LiFT-NCD on 11 datasets:

    ```bash
    sh train_liftncd.sh
    ```
