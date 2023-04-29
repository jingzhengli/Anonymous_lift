#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1 python train.py
# python train.py --dataset "ucf101" --clip_model 'RN50' --batch_size 128
# python train.py --dataset "dtd" --clip_model 'RN50' --batch_size 128
# python train.py --dataset "eurosat" --clip_model 'RN50' --batch_size 128
# python train.py --dataset "fgvc" --clip_model 'RN50' --batch_size 128
# python train.py --dataset "caltech101" --clip_model 'RN50' --batch_size 128
# python train.py --dataset "food101" --clip_model 'RN50' --batch_size 128
# python train.py --dataset "sun397" --clip_model 'RN50' --batch_size 128
# python train.py --dataset "stanford_cars" --clip_model 'RN50' --batch_size 128
# python train.py --dataset "oxford_flowers" --clip_model 'RN50' --batch_size 128
python train.py --dataset "ucf101" --batch_size 128 --shots 16
python train.py --dataset "dtd" --batch_size 128 --shots 16
python train.py --dataset "eurosat" --batch_size 128 --shots 16
python train.py --dataset "fgvc" --batch_size 128 --shots 16
python train.py --dataset "caltech101" --batch_size 128 --shots 16
python train.py --dataset "food101" --batch_size 128 --shots 16
python train.py --dataset "sun397" --batch_size 128 --shots 16
python train.py --dataset "stanford_cars" --batch_size 128 --shots 16
python train.py --dataset "oxford_flowers" --batch_size 128 --shots 16

python train.py --dataset "ucf101" --batch_size 128 --shots 8
python train.py --dataset "dtd" --batch_size 128 --shots 8
python train.py --dataset "eurosat" --batch_size 128 --shots 8
python train.py --dataset "fgvc" --batch_size 128 --shots 8
python train.py --dataset "caltech101" --batch_size 128 --shots 8
python train.py --dataset "food101" --batch_size 128 --shots 8
python train.py --dataset "sun397" --batch_size 128 --shots 8
python train.py --dataset "stanford_cars" --batch_size 128 --shots 8
python train.py --dataset "oxford_flowers" --batch_size 128 --shots 8

python train.py --dataset "ucf101" --batch_size 16 --shots 4
python train.py --dataset "dtd" --batch_size 16 --shots 4
python train.py --dataset "eurosat" --batch_size 16 --shots 4
python train.py --dataset "fgvc" --batch_size 16 --shots 4
python train.py --dataset "caltech101" --batch_size 16 --shots 4
python train.py --dataset "food101" --batch_size 128 --shots 4
python train.py --dataset "sun397" --batch_size 128 --shots 4
python train.py --dataset "stanford_cars" --batch_size 128 --shots 4
python train.py --dataset "oxford_flowers" --batch_size 128 --shots 4

python train.py --dataset "ucf101" --batch_size 16 --shots 6
python train.py --dataset "dtd" --batch_size 16 --shots 6
python train.py --dataset "eurosat" --batch_size 16 --shots 6
python train.py --dataset "fgvc" --batch_size 128 --shots 6
python train.py --dataset "caltech101" --batch_size 128 --shots 6
python train.py --dataset "food101" --batch_size 128 --shots 6
python train.py --dataset "sun397" --batch_size 128 --shots 6
python train.py --dataset "stanford_cars" --batch_size 128 --shots 6
python train.py --dataset "oxford_flowers" --batch_size 128 --shots 6

python train.py --dataset "ucf101" --batch_size 16 --shots 2
python train.py --dataset "dtd" --batch_size 16 --shots 2
python train.py --dataset "eurosat" --batch_size 16 --shots 2
python train.py --dataset "fgvc" --batch_size 16 --shots 2
python train.py --dataset "caltech101" --batch_size 16 --shots 2
python train.py --dataset "food101" --batch_size 16 --shots 2
python train.py --dataset "sun397" --batch_size 16 --shots 2
python train.py --dataset "stanford_cars" --batch_size 16 --shots 2
python train.py --dataset "oxford_flowers" --batch_size 16 --shots 2

python train.py --dataset "pets" --batch_size 16 --shots 2
python train.py --dataset "pets" --batch_size 16 --shots 4
python train.py --dataset "pets" --batch_size 16 --shots 6
python train.py --dataset "pets" --batch_size 16 --shots 8
python train.py --dataset "pets" --batch_size 16 --shots 16
python train.py --dataset "imagenet" --batch_size 128 --shots 16
python train.py --dataset "imagenet" --batch_size 128 --shots 8
python train.py --dataset "imagenet" --batch_size 128 --shots 6
python train.py --dataset "imagenet" --batch_size 128 --shots 4
python train.py --dataset "imagenet" --batch_size 128 --shots 2