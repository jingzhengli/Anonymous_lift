#!/usr/bin/env bash
python train_liftncd.py --dataset "ucf101" --batch_size 64 --shots 16
python train_liftncd.py --dataset "pets" --batch_size 64 --shots 16
python train_liftncd.py --dataset "dtd" --batch_size 64 --shots 16
python train_liftncd.py --dataset "eurosat" --batch_size 64 --shots 16
python train_liftncd.py --dataset "fgvc" --batch_size 64 --shots 16
python train_liftncd.py --dataset "caltech101" --batch_size 64 --shots 16
python train_liftncd.py --dataset "imagenet" --batch_size 64 --shots 16
python train_liftncd.py --dataset "sun397" --batch_size 64 --shots 16
python train_liftncd.py --dataset "oxford_flowers" --batch_size 64 --shots 16
python train_liftncd.py --dataset "stanford_cars" --batch_size 64 --shots 16
python train_liftncd.py --dataset "food101" --batch_size 64 --shots 16