#!/usr/bin/env bash
python train_flexmatch.py --dataset "ucf101" --batch_size 64 --shots 16
python train_flexmatch.py --dataset "pets" --batch_size 64 --shots 16
python train_flexmatch.py --dataset "dtd" --batch_size 64 --shots 16
python train_flexmatch.py --dataset "eurosat" --batch_size 64 --shots 16
python train_flexmatch.py --dataset "fgvc" --batch_size 64 --shots 16
python train_flexmatch.py --dataset "caltech101" --batch_size 64 --shots 16
python train_flexmatch.py --dataset "imagenet" --batch_size 64 --shots 16
python train_flexmatch.py --dataset "sun397" --batch_size 64 --shots 16
python train_flexmatch.py --dataset "oxford_flowers" --batch_size 64 --shots 16
python train_flexmatch.py --dataset "stanford_cars" --batch_size 64 --shots 16
python train_flexmatch.py --dataset "food101" --batch_size 64 --shots 16